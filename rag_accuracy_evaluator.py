import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_groq import ChatGroq
import numpy as np


@dataclass
class AccuracyMetrics:
    """Data class to store accuracy metrics"""
    overall_accuracy: float
    factual_accuracy: float
    grounding_accuracy: float
    relevance_accuracy: float
    total_questions: int
    correct_answers: int
    detailed_scores: List[Dict]


class RAGAccuracyEvaluator:
    """
    Professional RAG accuracy evaluator that calculates numerical accuracy scores
    """
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1,
            max_tokens=500
        )
        
        # Scoring thresholds
        self.accuracy_threshold = 0.7  # 70% threshold for considering answer "correct"
        
    def evaluate_single_answer(self, question: str, generated_answer: str, 
                             retrieved_contexts: List[str], expected_answer: str = None) -> Dict:
        """
        Evaluate a single Q&A pair for accuracy
        Returns numerical scores (0.0 to 1.0) for different aspects
        """
        try:
            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(
                question, generated_answer, retrieved_contexts, expected_answer
            )
            
            # Get LLM evaluation
            response = self.llm.invoke(evaluation_prompt)
            scores = self._parse_evaluation_response(response.content)
            
            # Calculate overall accuracy for this answer
            overall_score = np.mean([
                scores['factual_accuracy'],
                scores['grounding_accuracy'], 
                scores['relevance_accuracy']
            ])
            
            scores['overall_accuracy'] = overall_score
            scores['is_correct'] = overall_score >= self.accuracy_threshold
            
            return scores
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return self._default_scores()
    
    def _create_evaluation_prompt(self, question: str, answer: str, 
                                contexts: List[str], expected_answer: str = None) -> str:
        """Create structured evaluation prompt for numerical scoring"""
        
        context_text = "\n---\n".join(contexts) if contexts else "No context provided"
        expected_section = f"\nExpected Answer: {expected_answer}" if expected_answer else ""
        
        prompt = f"""
You are an expert evaluator. Score the generated answer on three aspects using a scale of 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_text}
{expected_section}

GENERATED ANSWER: {answer}

Evaluate and provide ONLY numerical scores (0.0 to 1.0) for:

1. FACTUAL_ACCURACY: Are the facts in the answer correct?
   - 1.0 = All facts are accurate
   - 0.5 = Some facts are correct, some wrong/unclear
   - 0.0 = Most facts are incorrect or misleading

2. GROUNDING_ACCURACY: Is the answer based on the provided context?
   - 1.0 = Answer is fully supported by context
   - 0.5 = Answer is partially supported by context
   - 0.0 = Answer contradicts or ignores context

3. RELEVANCE_ACCURACY: Does the answer directly address the question?
   - 1.0 = Directly and completely answers the question
   - 0.5 = Partially answers the question
   - 0.0 = Does not answer the question or is off-topic

Format your response EXACTLY as:
FACTUAL_ACCURACY: [score]
GROUNDING_ACCURACY: [score]  
RELEVANCE_ACCURACY: [score]
REASONING: [brief explanation]
"""
        return prompt
    
    def _parse_evaluation_response(self, response: str) -> Dict:
        """Parse LLM evaluation response to extract numerical scores"""
        scores = self._default_scores()
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if 'FACTUAL_ACCURACY:' in line:
                    scores['factual_accuracy'] = float(line.split(':')[1].strip())
                elif 'GROUNDING_ACCURACY:' in line:
                    scores['grounding_accuracy'] = float(line.split(':')[1].strip())
                elif 'RELEVANCE_ACCURACY:' in line:
                    scores['relevance_accuracy'] = float(line.split(':')[1].strip())
                elif 'REASONING:' in line:
                    scores['reasoning'] = line.split(':', 1)[1].strip()
        except Exception as e:
            print(f"Error parsing scores: {e}")
            
        return scores
    
    def _default_scores(self) -> Dict:
        """Return default scores in case of evaluation failure"""
        return {
            'factual_accuracy': 0.0,
            'grounding_accuracy': 0.0,
            'relevance_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'is_correct': False,
            'reasoning': 'Evaluation failed'
        }
    
    def evaluate_batch(self, qa_pairs: List[Dict], chatbot) -> AccuracyMetrics:
        """
        Evaluate multiple Q&A pairs and return overall accuracy metrics
        
        qa_pairs format: [
            {
                'question': 'How to harvest honey?',
                'expected_answer': 'Use a honey extractor...' (optional)
            }
        ]
        """
        detailed_scores = []
        correct_count = 0
        total_factual = []
        total_grounding = []
        total_relevance = []
        
        print(f"Evaluating {len(qa_pairs)} Q&A pairs...")
        
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair['question']
            expected_answer = qa_pair.get('expected_answer')
            
            print(f"Evaluating {i+1}/{len(qa_pairs)}: {question[:50]}...")
            
            # Get answer from your chatbot
            try:
                generated_answer, from_kb, confidence = chatbot.get_answer_with_confidence(question)
                
                # Get retrieved contexts
                contexts = []
                if hasattr(chatbot, 'vector_store') and chatbot.vector_store:
                    docs_with_scores = chatbot.vector_store.similarity_search_with_score(question, k=3)
                    contexts = [doc.page_content for doc, score in docs_with_scores]
                
                # Evaluate this answer
                scores = self.evaluate_single_answer(
                    question, generated_answer, contexts, expected_answer
                )
                
                # Add metadata
                scores.update({
                    'question': question,
                    'generated_answer': generated_answer,
                    'expected_answer': expected_answer,
                    'from_knowledge_base': from_kb,
                    'confidence': confidence
                })
                
                detailed_scores.append(scores)
                
                # Collect metrics
                if scores['is_correct']:
                    correct_count += 1
                
                total_factual.append(scores['factual_accuracy'])
                total_grounding.append(scores['grounding_accuracy'])
                total_relevance.append(scores['relevance_accuracy'])
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                continue
        
        # Calculate overall metrics
        overall_accuracy = correct_count / len(qa_pairs) if qa_pairs else 0
        avg_factual = np.mean(total_factual) if total_factual else 0
        avg_grounding = np.mean(total_grounding) if total_grounding else 0
        avg_relevance = np.mean(total_relevance) if total_relevance else 0
        
        metrics = AccuracyMetrics(
            overall_accuracy=overall_accuracy,
            factual_accuracy=avg_factual,
            grounding_accuracy=avg_grounding,
            relevance_accuracy=avg_relevance,
            total_questions=len(qa_pairs),
            correct_answers=correct_count,
            detailed_scores=detailed_scores
        )
        
        return metrics
    
    def print_accuracy_report(self, metrics: AccuracyMetrics):
        """Print a professional accuracy report"""
        print("\n" + "="*60)
        print("           RAG SYSTEM ACCURACY REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   • Overall Accuracy:     {metrics.overall_accuracy:.1%}")
        print(f"   • Correct Answers:      {metrics.correct_answers}/{metrics.total_questions}")
        
        print(f"\n📋 DETAILED SCORES:")
        print(f"   • Factual Accuracy:     {metrics.factual_accuracy:.1%}")
        print(f"   • Grounding Accuracy:   {metrics.grounding_accuracy:.1%}")
        print(f"   • Relevance Accuracy:   {metrics.relevance_accuracy:.1%}")
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        if metrics.overall_accuracy >= 0.8:
            print("   ✅ EXCELLENT - System performs very well")
        elif metrics.overall_accuracy >= 0.6:
            print("   ⚠️  GOOD - System performs adequately")
        else:
            print("   ❌ NEEDS IMPROVEMENT - System requires optimization")
            
        print("\n" + "="*60)
    
    def save_detailed_report(self, metrics: AccuracyMetrics, filepath: str):
        """Save detailed evaluation report to JSON file"""
        report = {
            'summary': {
                'overall_accuracy': metrics.overall_accuracy,
                'factual_accuracy': metrics.factual_accuracy,
                'grounding_accuracy': metrics.grounding_accuracy,
                'relevance_accuracy': metrics.relevance_accuracy,
                'total_questions': metrics.total_questions,
                'correct_answers': metrics.correct_answers
            },
            'detailed_scores': metrics.detailed_scores
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detailed report saved to: {filepath}")


def create_sample_test_questions() -> List[Dict]:
    """Create sample test questions for beekeeping domain"""
    return [
        {
            'question': 'What does "organic honey" mean',
            'expected_answer': 'Honey produced without synthetic chemicals and from certified organic flora.'
        },
        {
            'question': 'What acids are found in honey and which is the most prevalent?',
            'expected_answer': 'Gluconic acid is the major acid in honey, arising from dextrose through glucose oxidase enzyme action. Other acids include formic, acetic, but yric, lactic, oxalic, succinic, tartaric, maleic, pyruvic, pyroglutamic, α-ketoglutaric, glycollic, citric, malic, 2- or 3-phosphoglyceric acid, α- or β-glycerophosphate , and glucose 6-phosphate. The acids account for less than 0.5% of solids but contribute to flavor and stability.'
        },
        {
            'question': 'Describe the role of enzymes in honey and their importance.',
            'expected_answer': "Honey contains several important enzymes added by bees during processing. Invertase breaks down sucrose into glucose and fructose, making honey more digestible. Glucose oxidase produces hydrogen peroxide and gluconic acid, contributing to honey's antimicrobial properties and acidic pH. Catalase breaks down hydrogen peroxide to prevent excessive accumulation. Diastase helps break down starches and is used as an indicator of honey freshness and heat treatment. These enzymes are heat-sensitive and can be destroyed by excessive processing, which is why raw honey is often preferred for its full enzymatic activity."
        },
        {
            'question': 'Explain the pH range of honey and its significance',
            'expected_answer': "Honey has an acidic pH ranging from 3.2 to 4.5, with an average around 3.9. This acidity is primarily due to gluconic acid produced by glucose oxidase enzyme activity and organic acids from floral sources. The low pH contributes to honey's antimicrobial properties, stability against spoilage, flavor profile, and helps preserve its quality during storage. pH also serves as a quality parameter for honey authentication."
        },
        {
            'question': 'What is the role of honey in the Mediterranean diet?',
            'expected_answer': "In the Mediterranean diet, honey is valued for its natural sweetness and health benefits, often used as a sweetener in various dishes, contributing to the diet's emphasis on whole, unprocessed foods"
        }
    ]


# Integration function for your main chatbot
def run_accuracy_evaluation(chatbot, groq_api_key: str, 
                          test_questions: List[Dict] = None,
                          save_report: bool = True) -> AccuracyMetrics:
    """
    Main function to run accuracy evaluation on your RAG chatbot
    """
    if test_questions is None:
        test_questions = create_sample_test_questions()
    
    # Initialize evaluator
    evaluator = RAGAccuracyEvaluator(groq_api_key)
    
    # Run evaluation
    print("🔍 Starting RAG Accuracy Evaluation...")
    metrics = evaluator.evaluate_batch(test_questions, chatbot)
    
    # Print report
    evaluator.print_accuracy_report(metrics)
    
    # Save detailed report if requested
    if save_report:
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"rag_accuracy_report_{timestamp}.json"
        evaluator.save_detailed_report(metrics, report_file)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("RAG Accuracy Evaluator - Ready for integration")
    print("Import this module and call run_accuracy_evaluation(chatbot, groq_api_key)")