"""
RAG Evaluation Module
Author: Your Name
Description: Comprehensive evaluation framework for RAG systems
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics

@dataclass
class EvaluationMetrics:
    """Data class to store comprehensive evaluation metrics"""
    question: str
    generated_answer: str
    reference_answer: Optional[str] = None
    retrieved_contexts: List[str] = None
    
    # Accuracy Metrics (0.0 - 1.0)
    semantic_similarity: float = 0.0
    factual_accuracy: float = 0.0
    answer_relevance: float = 0.0
    
    # Quality Metrics (0.0 - 1.0)
    completeness: float = 0.0
    clarity: float = 0.0
    coherence: float = 0.0
    
    # Conformity Metrics (0.0 - 1.0)
    context_adherence: float = 0.0
    hallucination_score: float = 0.0  # 0.0 = no hallucination, 1.0 = high hallucination
    knowledge_base_alignment: float = 0.0
    
    # Retrieval Metrics (0.0 - 1.0)
    context_precision: float = 0.0
    context_recall: float = 0.0
    retrieval_quality: float = 0.0
    
    # Overall Scores
    accuracy_score: float = 0.0
    quality_score: float = 0.0
    conformity_score: float = 0.0
    overall_score: float = 0.0
    
    # Metadata
    evaluation_timestamp: str = None
    confidence_level: str = "medium"
    
    def __post_init__(self):
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization"""
        return asdict(self)

class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self, llm: ChatGroq, debug: bool = False):
        """
        Initialize the RAG evaluator
        
        Args:
            llm: ChatGroq language model instance
            debug: Enable debug mode for verbose output
        """
        self.llm = llm
        self.debug = debug
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Evaluation templates
        self._init_templates()
    
    def _init_templates(self):
        """Initialize evaluation prompt templates"""
        
        self.factual_accuracy_template = PromptTemplate(
            template="""You are an expert fact-checker for beekeeping and honey production knowledge.

Generated Answer: {generated_answer}

Source Contexts: {contexts}

Task: Rate the factual accuracy of the generated answer against the source contexts.

Evaluation Criteria:
- 1.0: All facts are accurate and properly supported
- 0.8: Mostly accurate with minor issues  
- 0.6: Generally accurate but some notable errors
- 0.4: Mixed accuracy with significant errors
- 0.2: Mostly inaccurate information
- 0.0: Completely inaccurate or unsupported

Return only a decimal number (0.0-1.0):""",
            input_variables=["generated_answer", "contexts"]
        )
        
        self.relevance_template = PromptTemplate(
            template="""Evaluate how well this answer addresses the specific question asked.

Question: {question}
Generated Answer: {generated_answer}

Rate answer relevance (0.0-1.0):
- 1.0: Directly and completely answers the question
- 0.8: Good answer with minor irrelevance
- 0.6: Partially answers the question
- 0.4: Somewhat related but misses key points
- 0.2: Barely addresses the question
- 0.0: Completely irrelevant

Return only a decimal number (0.0-1.0):""",
            input_variables=["question", "generated_answer"]
        )
        
        self.completeness_template = PromptTemplate(
            template="""Evaluate the completeness of this answer given the available information.

Question: {question}
Generated Answer: {generated_answer}
Available Context: {contexts}

Rate completeness (0.0-1.0):
- 1.0: Comprehensive, covers all important aspects
- 0.8: Good coverage with minor gaps
- 0.6: Adequate but missing some key points
- 0.4: Incomplete with notable gaps
- 0.2: Very incomplete
- 0.0: Severely incomplete or unhelpful

Return only a decimal number (0.0-1.0):""",
            input_variables=["question", "generated_answer", "contexts"]
        )
        
        self.context_adherence_template = PromptTemplate(
            template="""Evaluate how well the answer sticks to information from the provided contexts.

Generated Answer: {generated_answer}
Source Contexts: {contexts}

Rate context adherence (0.0-1.0):
- 1.0: Answer uses only information from contexts
- 0.8: Mostly from contexts with minimal external info
- 0.6: Good use of contexts but some external information
- 0.4: Mixed use of context and external knowledge
- 0.2: Limited use of provided contexts
- 0.0: Ignores contexts, uses mostly external information

Return only a decimal number (0.0-1.0):""",
            input_variables=["generated_answer", "contexts"]
        )
        
        self.hallucination_template = PromptTemplate(
            template="""Detect potential hallucinations in this answer compared to the source contexts.

Generated Answer: {generated_answer}
Source Contexts: {contexts}

Rate hallucination level (0.0-1.0):
- 0.0: No hallucinations, all information is supported
- 0.2: Minor unsupported details
- 0.4: Some notable unsupported claims
- 0.6: Multiple unsupported or invented facts
- 0.8: Significant fabricated information  
- 1.0: Mostly hallucinated content

Return only a decimal number (0.0-1.0):""",
            input_variables=["generated_answer", "contexts"]
        )
    
    def _extract_score(self, response_text: str, default: float = 0.0) -> float:
        """Extract numerical score from LLM response"""
        try:
            # Look for decimal numbers
            score_match = re.search(r'(\d+\.?\d*)', response_text.strip())
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))
            return default
        except Exception as e:
            if self.debug:
                print(f"Score extraction error: {e}")
            return default
    
    def _safe_llm_call(self, template: PromptTemplate, **kwargs) -> float:
        """Safely call LLM with error handling"""
        try:
            prompt = template.format(**kwargs)
            response = self.llm.invoke(prompt)
            return self._extract_score(response.content)
        except Exception as e:
            if self.debug:
                print(f"LLM call error: {e}")
            return 0.0
    
    def evaluate_semantic_similarity(self, generated: str, reference: str) -> float:
        """Compute semantic similarity using TF-IDF cosine similarity"""
        try:
            if not reference or not generated:
                return 0.0
            
            texts = [generated.lower(), reference.lower()]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            if self.debug:
                print(f"Semantic similarity error: {e}")
            return 0.0
    
    def evaluate_factual_accuracy(self, generated: str, contexts: List[str]) -> float:
        """Evaluate factual accuracy against source contexts"""
        if not contexts:
            return 0.0
        
        contexts_text = "\n---\n".join(contexts[:3])  # Limit context length
        return self._safe_llm_call(
            self.factual_accuracy_template,
            generated_answer=generated,
            contexts=contexts_text
        )
    
    def evaluate_answer_relevance(self, question: str, generated: str) -> float:
        """Evaluate answer relevance to the question"""
        return self._safe_llm_call(
            self.relevance_template,
            question=question,
            generated_answer=generated
        )
    
    def evaluate_completeness(self, question: str, generated: str, contexts: List[str]) -> float:
        """Evaluate answer completeness"""
        contexts_text = "\n---\n".join(contexts[:3]) if contexts else "No context available"
        return self._safe_llm_call(
            self.completeness_template,
            question=question,
            generated_answer=generated,
            contexts=contexts_text
        )
    
    def evaluate_context_adherence(self, generated: str, contexts: List[str]) -> float:
        """Evaluate how well answer adheres to provided contexts"""
        if not contexts:
            return 0.0
        
        contexts_text = "\n---\n".join(contexts[:3])
        return self._safe_llm_call(
            self.context_adherence_template,
            generated_answer=generated,
            contexts=contexts_text
        )
    
    def evaluate_hallucination(self, generated: str, contexts: List[str]) -> float:
        """Detect hallucinations in the generated answer"""
        if not contexts:
            return 1.0  # High hallucination if no context
        
        contexts_text = "\n---\n".join(contexts[:3])
        return self._safe_llm_call(
            self.hallucination_template,
            generated_answer=generated,
            contexts=contexts_text
        )
    
    def evaluate_clarity_coherence(self, generated: str) -> Tuple[float, float]:
        """Evaluate clarity and coherence using simple heuristics"""
        try:
            # Simple clarity metrics
            sentences = generated.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Normalize sentence length (ideal: 10-20 words)
            clarity = 1.0 - min(1.0, abs(avg_sentence_length - 15) / 15)
            
            # Simple coherence: check for transition words and logical flow
            transition_words = ['however', 'therefore', 'moreover', 'additionally', 'furthermore', 
                             'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example']
            
            transition_count = sum(1 for word in transition_words if word in generated.lower())
            coherence = min(1.0, transition_count / max(1, len(sentences)))
            
            return max(0.0, clarity), max(0.0, coherence)
            
        except Exception as e:
            if self.debug:
                print(f"Clarity/coherence evaluation error: {e}")
            return 0.5, 0.5  # Default moderate scores
    
    def evaluate_retrieval_quality(self, question: str, contexts: List[str], 
                                 reference_contexts: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate retrieval quality metrics"""
        try:
            if not contexts:
                return {"precision": 0.0, "recall": 0.0, "quality": 0.0}
            
            # Simple relevance check using keyword overlap
            question_words = set(question.lower().split())
            
            relevant_contexts = 0
            for context in contexts:
                context_words = set(context.lower().split())
                overlap = len(question_words.intersection(context_words))
                if overlap >= 2:  # At least 2 words overlap
                    relevant_contexts += 1
            
            precision = relevant_contexts / len(contexts) if contexts else 0.0
            recall = 1.0 if relevant_contexts > 0 else 0.0  # Simplified recall
            
            # Overall retrieval quality
            quality = (precision + recall) / 2 if (precision + recall) > 0 else 0.0
            
            return {
                "precision": precision,
                "recall": recall,
                "quality": quality
            }
            
        except Exception as e:
            if self.debug:
                print(f"Retrieval quality error: {e}")
            return {"precision": 0.0, "recall": 0.0, "quality": 0.0}
    
    def comprehensive_evaluate(self, question: str, generated_answer: str, 
                             retrieved_contexts: List[str],
                             reference_answer: Optional[str] = None,
                             confidence_threshold: float = 0.7) -> EvaluationMetrics:
        """
        Perform comprehensive evaluation of RAG response
        
        Args:
            question: The input question
            generated_answer: RAG system's generated answer
            retrieved_contexts: List of retrieved context documents
            reference_answer: Optional ground truth answer
            confidence_threshold: Threshold for determining confidence level
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        
        if self.debug:
            print(f"Evaluating question: {question[:50]}...")
        
        # Initialize metrics
        metrics = EvaluationMetrics(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            retrieved_contexts=retrieved_contexts
        )
        
        # 1. Accuracy Metrics
        if reference_answer:
            metrics.semantic_similarity = self.evaluate_semantic_similarity(
                generated_answer, reference_answer
            )
        
        metrics.factual_accuracy = self.evaluate_factual_accuracy(
            generated_answer, retrieved_contexts
        )
        
        metrics.answer_relevance = self.evaluate_answer_relevance(
            question, generated_answer
        )
        
        # 2. Quality Metrics
        metrics.completeness = self.evaluate_completeness(
            question, generated_answer, retrieved_contexts
        )
        
        clarity, coherence = self.evaluate_clarity_coherence(generated_answer)
        metrics.clarity = clarity
        metrics.coherence = coherence
        
        # 3. Conformity Metrics
        metrics.context_adherence = self.evaluate_context_adherence(
            generated_answer, retrieved_contexts
        )
        
        metrics.hallucination_score = self.evaluate_hallucination(
            generated_answer, retrieved_contexts
        )
        
        metrics.knowledge_base_alignment = (
            metrics.context_adherence + (1.0 - metrics.hallucination_score)
        ) / 2
        
        # 4. Retrieval Metrics
        retrieval_scores = self.evaluate_retrieval_quality(question, retrieved_contexts)
        metrics.context_precision = retrieval_scores["precision"]
        metrics.context_recall = retrieval_scores["recall"]
        metrics.retrieval_quality = retrieval_scores["quality"]
        
        # 5. Compute Overall Scores
        metrics.accuracy_score = statistics.mean([
            metrics.factual_accuracy,
            metrics.answer_relevance,
            metrics.semantic_similarity if reference_answer else 0.5
        ])
        
        metrics.quality_score = statistics.mean([
            metrics.completeness,
            metrics.clarity,
            metrics.coherence
        ])
        
        metrics.conformity_score = statistics.mean([
            metrics.context_adherence,
            1.0 - metrics.hallucination_score,  # Invert hallucination
            metrics.knowledge_base_alignment
        ])
        
        # Overall score (weighted average)
        weights = {
            'accuracy': 0.4,
            'quality': 0.3,
            'conformity': 0.3
        }
        
        metrics.overall_score = (
            weights['accuracy'] * metrics.accuracy_score +
            weights['quality'] * metrics.quality_score +
            weights['conformity'] * metrics.conformity_score
        )
        
        # Determine confidence level
        if metrics.overall_score >= confidence_threshold:
            metrics.confidence_level = "high"
        elif metrics.overall_score >= confidence_threshold - 0.2:
            metrics.confidence_level = "medium"
        else:
            metrics.confidence_level = "low"
        
        if self.debug:
            print(f"Overall score: {metrics.overall_score:.3f} ({metrics.confidence_level} confidence)")
        
        return metrics
    
    def batch_evaluate(self, evaluation_data: List[Dict], 
                      save_results: bool = True,
                      output_file: str = "rag_evaluation_results.json") -> List[EvaluationMetrics]:
        """
        Evaluate multiple question-answer pairs in batch
        
        Args:
            evaluation_data: List of dicts with 'question', 'generated_answer', 
                           'retrieved_contexts', and optionally 'reference_answer'
            save_results: Whether to save results to file
            output_file: Output file name for results
            
        Returns:
            List of EvaluationMetrics objects
        """
        results = []
        
        print(f"Starting batch evaluation of {len(evaluation_data)} items...")
        
        for i, item in enumerate(evaluation_data):
            if self.debug or (i + 1) % 10 == 0:
                print(f"Evaluating item {i + 1}/{len(evaluation_data)}")
            
            try:
                metrics = self.comprehensive_evaluate(
                    question=item.get('question', ''),
                    generated_answer=item.get('generated_answer', ''),
                    retrieved_contexts=item.get('retrieved_contexts', []),
                    reference_answer=item.get('reference_answer')
                )
                results.append(metrics)
                
            except Exception as e:
                print(f"Error evaluating item {i + 1}: {e}")
                continue
        
        if save_results and results:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[EvaluationMetrics], filename: str):
        """Save evaluation results to JSON file"""
        try:
            results_dict = [result.to_dict() for result in results]
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def generate_report(self, results: List[EvaluationMetrics]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Calculate aggregate statistics
        metrics_stats = {}
        metric_names = [
            'semantic_similarity', 'factual_accuracy', 'answer_relevance',
            'completeness', 'clarity', 'coherence', 'context_adherence',
            'hallucination_score', 'knowledge_base_alignment',
            'context_precision', 'context_recall', 'retrieval_quality',
            'accuracy_score', 'quality_score', 'conformity_score', 'overall_score'
        ]
        
        for metric in metric_names:
            values = [getattr(result, metric) for result in results if hasattr(result, metric)]
            if values:
                metrics_stats[metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
        
        # Confidence level distribution
        confidence_dist = Counter([result.confidence_level for result in results])
        
        # Performance categories
        high_performers = [r for r in results if r.overall_score >= 0.8]
        low_performers = [r for r in results if r.overall_score < 0.5]
        
        report = {
            'summary': {
                'total_evaluations': len(results),
                'average_overall_score': metrics_stats.get('overall_score', {}).get('mean', 0.0),
                'confidence_distribution': dict(confidence_dist),
                'high_performers': len(high_performers),
                'low_performers': len(low_performers)
            },
            'detailed_metrics': metrics_stats,
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[EvaluationMetrics]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        if not results:
            return recommendations
        
        # Calculate average scores
        avg_factual = statistics.mean([r.factual_accuracy for r in results])
        avg_coherence = statistics.mean([r.coherence for r in results])
        avg_completeness = statistics.mean([r.completeness for r in results])
        avg_hallucination = statistics.mean([r.hallucination_score for r in results])
        avg_context_adherence = statistics.mean([r.context_adherence for r in results])
        
        if avg_factual < 0.6:
            recommendations.append("Improve factual accuracy by enhancing knowledge base quality and fact-checking mechanisms")
        
        if avg_coherence < 0.6:
            recommendations.append("Enhance response coherence through better prompt engineering and response structure")
        
        if avg_completeness < 0.6:
            recommendations.append("Improve answer completeness by retrieving more relevant contexts and better synthesis")
        
        if avg_hallucination > 0.4:
            recommendations.append("Reduce hallucinations by strengthening context adherence and limiting creative responses")
        
        if avg_context_adherence < 0.6:
            recommendations.append("Improve context adherence through better prompt templates and stricter instructions")
        
        return recommendations

# Convenience functions for easy integration
def quick_evaluate(llm: ChatGroq, question: str, answer: str, 
                  contexts: List[str], reference: str = None) -> EvaluationMetrics:
    """Quick evaluation function for single question-answer pair"""
    evaluator = RAGEvaluator(llm)
    return evaluator.comprehensive_evaluate(question, answer, contexts, reference)

def create_test_set(jsonl_data: List[Dict], num_samples: int = 50) -> List[Dict]:
    """Create test set from JSONL knowledge base for evaluation"""
    import random
    
    # Sample questions from knowledge base
    samples = random.sample(jsonl_data, min(num_samples, len(jsonl_data)))
    
    test_set = []
    for sample in samples:
        test_item = {
            'question': sample.get('instruction', ''),
            'reference_answer': sample.get('output', ''),
            'reference_context': sample.get('input', '')
        }
        test_set.append(test_item)
    
    return test_set