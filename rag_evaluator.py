import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

class RAGAnswerEvaluator:
    """
    Comprehensive evaluation system for RAG-based chatbot answers.
    Evaluates faithfulness, relevance, accuracy, and hallucination detection.
    """
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        # Initialize evaluation LLM (using different model for objectivity)
        self.eval_llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",  # Use same or different model
            temperature=0.0,  # Minimize randomness for consistent evaluation
            max_tokens=500
        )
        
        # Initialize sentence transformer for semantic similarity
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Evaluation metrics storage
        self.evaluation_history = []
        
    def evaluate_faithfulness(self, question: str, answer: str, retrieved_contexts: List[str]) -> Dict:
        """
        Evaluate if the answer is faithful to the retrieved context.
        Uses LLM-based evaluation for nuanced assessment.
        """
        
        # Combine all retrieved contexts
        combined_context = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)])
        
        faithfulness_prompt = PromptTemplate(
            template="""
You are an expert evaluator assessing whether an AI assistant's answer is faithful to the provided context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate the answer on the following criteria:

1. FAITHFULNESS (0-5): Does the answer only use information present in the context?
   - 5: Answer is completely grounded in context, no external information
   - 4: Answer is mostly grounded, minor acceptable inferences
   - 3: Answer is partially grounded, some unsupported claims
   - 2: Answer has significant unsupported information
   - 1: Answer contradicts the context or adds major unsupported claims
   - 0: Answer is completely unrelated to context

2. COMPLETENESS (0-5): Does the answer address the question using available context?
   - 5: Fully addresses question with all relevant context information
   - 4: Addresses most aspects with good use of context
   - 3: Addresses some aspects, missing some relevant context
   - 2: Partially addresses question, underutilizes context
   - 1: Minimally addresses question
   - 0: Does not address the question

3. ACCURACY (0-5): Is the information in the answer factually correct based on context?
   - 5: All information is accurate according to context
   - 4: Mostly accurate with minor issues
   - 3: Generally accurate with some errors
   - 2: Several inaccuracies present
   - 1: Mostly inaccurate information
   - 0: Completely inaccurate

Provide your evaluation in this exact format:
FAITHFULNESS_SCORE: [0-5]
COMPLETENESS_SCORE: [0-5]
ACCURACY_SCORE: [0-5]
REASONING: [Brief explanation of scores]
HALLUCINATION_DETECTED: [YES/NO]
SPECIFIC_ISSUES: [List any specific problems or "None"]
            """,
            input_variables=["question", "context", "answer"]
        )
        
        try:
            eval_response = self.eval_llm.invoke(
                faithfulness_prompt.format(
                    question=question,
                    context=combined_context,
                    answer=answer
                )
            )
            
            # Parse the evaluation response
            return self._parse_faithfulness_response(eval_response.content)
            
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return {
                "faithfulness_score": 0,
                "completeness_score": 0,
                "accuracy_score": 0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "hallucination_detected": True,
                "specific_issues": ["Evaluation system error"]
            }
    
    def _parse_faithfulness_response(self, response: str) -> Dict:
        """Parse the LLM evaluation response into structured data."""
        try:
            # Extract scores using regex
            faithfulness_match = re.search(r'FAITHFULNESS_SCORE:\s*(\d)', response)
            completeness_match = re.search(r'COMPLETENESS_SCORE:\s*(\d)', response)
            accuracy_match = re.search(r'ACCURACY_SCORE:\s*(\d)', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            hallucination_match = re.search(r'HALLUCINATION_DETECTED:\s*(YES|NO)', response)
            issues_match = re.search(r'SPECIFIC_ISSUES:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            
            return {
                "faithfulness_score": int(faithfulness_match.group(1)) if faithfulness_match else 0,
                "completeness_score": int(completeness_match.group(1)) if completeness_match else 0,
                "accuracy_score": int(accuracy_match.group(1)) if accuracy_match else 0,
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided",
                "hallucination_detected": hallucination_match.group(1) == "YES" if hallucination_match else True,
                "specific_issues": [issues_match.group(1).strip()] if issues_match else ["Unknown issues"]
            }
        except Exception as e:
            return {
                "faithfulness_score": 0,
                "completeness_score": 0,
                "accuracy_score": 0,
                "reasoning": f"Failed to parse evaluation: {str(e)}",
                "hallucination_detected": True,
                "specific_issues": ["Response parsing error"]
            }
    
    def evaluate_semantic_similarity(self, question: str, answer: str, retrieved_contexts: List[str]) -> Dict:
        """
        Calculate semantic similarity between answer and retrieved contexts.
        """
        try:
            if not retrieved_contexts:
                return {
                    "max_context_similarity": 0.0,
                    "avg_context_similarity": 0.0,
                    "question_answer_similarity": 0.0
                }
            
            # Get embeddings
            answer_embedding = self.similarity_model.encode([answer])
            context_embeddings = self.similarity_model.encode(retrieved_contexts)
            question_embedding = self.similarity_model.encode([question])
            
            # Calculate similarities
            context_similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
            question_similarity = cosine_similarity(answer_embedding, question_embedding)[0][0]
            
            return {
                "max_context_similarity": float(np.max(context_similarities)),
                "avg_context_similarity": float(np.mean(context_similarities)),
                "question_answer_similarity": float(question_similarity),
                "individual_context_similarities": context_similarities.tolist()
            }
            
        except Exception as e:
            print(f"Error in semantic similarity evaluation: {e}")
            return {
                "max_context_similarity": 0.0,
                "avg_context_similarity": 0.0,
                "question_answer_similarity": 0.0,
                "error": str(e)
            }
    
    def evaluate_answer_quality(self, question: str, answer: str) -> Dict:
        """
        Evaluate overall answer quality without context dependency.
        """
        quality_prompt = PromptTemplate(
            template="""
Evaluate the quality of this AI assistant's answer:

QUESTION: {question}
ANSWER: {answer}

Rate the answer on these criteria (0-5 scale):

1. CLARITY: How clear and well-structured is the answer?
2. RELEVANCE: How well does it address the question asked?
3. COMPLETENESS: Does it provide a comprehensive response?
4. HELPFULNESS: How useful would this be to the person asking?

Provide scores in this format:
CLARITY_SCORE: [0-5]
RELEVANCE_SCORE: [0-5]
COMPLETENESS_SCORE: [0-5]
HELPFULNESS_SCORE: [0-5]
OVERALL_QUALITY: [0-5]
FEEDBACK: [Brief constructive feedback]
            """,
            input_variables=["question", "answer"]
        )
        
        try:
            eval_response = self.eval_llm.invoke(
                quality_prompt.format(question=question, answer=answer)
            )
            return self._parse_quality_response(eval_response.content)
        except Exception as e:
            return {
                "clarity_score": 0,
                "relevance_score": 0,
                "completeness_score": 0,
                "helpfulness_score": 0,
                "overall_quality": 0,
                "feedback": f"Quality evaluation failed: {str(e)}"
            }
    
    def _parse_quality_response(self, response: str) -> Dict:
        """Parse quality evaluation response."""
        try:
            clarity_match = re.search(r'CLARITY_SCORE:\s*(\d)', response)
            relevance_match = re.search(r'RELEVANCE_SCORE:\s*(\d)', response)
            completeness_match = re.search(r'COMPLETENESS_SCORE:\s*(\d)', response)
            helpfulness_match = re.search(r'HELPFULNESS_SCORE:\s*(\d)', response)
            overall_match = re.search(r'OVERALL_QUALITY:\s*(\d)', response)
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)$', response, re.DOTALL)
            
            return {
                "clarity_score": int(clarity_match.group(1)) if clarity_match else 0,
                "relevance_score": int(relevance_match.group(1)) if relevance_match else 0,
                "completeness_score": int(completeness_match.group(1)) if completeness_match else 0,
                "helpfulness_score": int(helpfulness_match.group(1)) if helpfulness_match else 0,
                "overall_quality": int(overall_match.group(1)) if overall_match else 0,
                "feedback": feedback_match.group(1).strip() if feedback_match else "No feedback provided"
            }
        except Exception as e:
            return {
                "clarity_score": 0,
                "relevance_score": 0,
                "completeness_score": 0,
                "helpfulness_score": 0,
                "overall_quality": 0,
                "feedback": f"Failed to parse quality evaluation: {str(e)}"
            }
    
    def comprehensive_evaluation(self, 
                                question: str, 
                                answer: str, 
                                retrieved_contexts: List[str],
                                confidence_score: float = None,
                                from_knowledge_base: bool = None) -> Dict:
        """
        Perform comprehensive evaluation of a RAG answer.
        """
        
        print(f"Evaluating: {question[:50]}...")
        
        # Run all evaluations
        faithfulness_eval = self.evaluate_faithfulness(question, answer, retrieved_contexts)
        similarity_eval = self.evaluate_semantic_similarity(question, answer, retrieved_contexts)
        quality_eval = self.evaluate_answer_quality(question, answer)
        
        # Compile comprehensive results
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "retrieved_contexts_count": len(retrieved_contexts),
            "confidence_score": confidence_score,
            "from_knowledge_base": from_knowledge_base,
            
            # Faithfulness metrics
            **faithfulness_eval,
            
            # Similarity metrics
            **similarity_eval,
            
            # Quality metrics
            **quality_eval,
            
            # Composite scores
            "composite_faithfulness": (faithfulness_eval["faithfulness_score"] + 
                                     faithfulness_eval["accuracy_score"]) / 2,
            "composite_quality": (quality_eval["clarity_score"] + 
                                quality_eval["relevance_score"] + 
                                quality_eval["helpfulness_score"]) / 3
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def evaluate_test_set(self, test_cases: List[Dict]) -> pd.DataFrame:
        """
        Evaluate a set of test cases.
        Expected format: [{"question": str, "answer": str, "contexts": List[str], ...}, ...]
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = self.comprehensive_evaluation(
                question=test_case["question"],
                answer=test_case["answer"],
                retrieved_contexts=test_case.get("contexts", []),
                confidence_score=test_case.get("confidence", None),
                from_knowledge_base=test_case.get("from_kb", None)
            )
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, results_df: pd.DataFrame = None) -> Dict:
        """Generate comprehensive evaluation report."""
        if results_df is None:
            if not self.evaluation_history:
                return {"error": "No evaluation data available"}
            results_df = pd.DataFrame(self.evaluation_history)
        
        report = {
            "summary": {
                "total_evaluations": len(results_df),
                "avg_faithfulness_score": results_df["faithfulness_score"].mean(),
                "avg_accuracy_score": results_df["accuracy_score"].mean(),
                "avg_completeness_score": results_df["completeness_score"].mean(),
                "hallucination_rate": (results_df["hallucination_detected"] == True).mean(),
                "avg_semantic_similarity": results_df["max_context_similarity"].mean(),
                "avg_overall_quality": results_df["overall_quality"].mean()
            },
            "score_distributions": {
                "faithfulness_distribution": results_df["faithfulness_score"].value_counts().to_dict(),
                "accuracy_distribution": results_df["accuracy_score"].value_counts().to_dict(),
                "quality_distribution": results_df["overall_quality"].value_counts().to_dict()
            },
            "problematic_cases": []
        }
        
        # Identify problematic cases
        problematic = results_df[
            (results_df["faithfulness_score"] < 3) | 
            (results_df["hallucination_detected"] == True) |
            (results_df["accuracy_score"] < 3)
        ]
        
        for _, case in problematic.head(10).iterrows():
            report["problematic_cases"].append({
                "question": case["question"][:100] + "..." if len(case["question"]) > 100 else case["question"],
                "faithfulness_score": case["faithfulness_score"],
                "accuracy_score": case["accuracy_score"],
                "hallucination_detected": case["hallucination_detected"],
                "issues": case["specific_issues"]
            })
        
        return report
    
    def save_evaluation_results(self, filename: str, results_df: pd.DataFrame = None):
        """Save evaluation results to file."""
        if results_df is None:
            results_df = pd.DataFrame(self.evaluation_history)
        
        # Save as CSV
        csv_filename = filename.replace('.json', '.csv')
        results_df.to_csv(csv_filename, index=False)
        
        # Save as JSON with report
        report = self.generate_evaluation_report(results_df)
        with open(filename, 'w') as f:
            json.dump({
                "evaluation_report": report,
                "raw_data": results_df.to_dict('records')
            }, f, indent=2, default=str)
        
        print(f"Results saved to {csv_filename} and {filename}")

# Integration helper functions
def integrate_with_chatbot(chatbot_instance, evaluator: RAGAnswerEvaluator):
    """
    Modify the chatbot to include automatic evaluation.
    This monkey-patches the existing chatbot's get_answer_with_confidence method.
    """
    original_method = chatbot_instance.get_answer_with_confidence
    
    def enhanced_get_answer_with_confidence(question: str, session_id: str = "default"):
        # Get original answer
        answer, from_kb, confidence = original_method(question, session_id)
        
        # Get retrieved contexts for evaluation
        try:
            if chatbot_instance.vector_store and from_kb:
                docs_with_scores = chatbot_instance.vector_store.similarity_search_with_score(question, k=3)
                contexts = [doc.page_content for doc, score in docs_with_scores]
            else:
                contexts = []
            
            # Perform evaluation (async to not slow down chat)
            evaluation_result = evaluator.comprehensive_evaluation(
                question=question,
                answer=answer,
                retrieved_contexts=contexts,
                confidence_score=confidence,
                from_knowledge_base=from_kb
            )
            
            # Log evaluation (optional)
            print(f"Evaluation - Faithfulness: {evaluation_result['faithfulness_score']}/5, "
                  f"Quality: {evaluation_result['overall_quality']}/5, "
                  f"Hallucination: {'Yes' if evaluation_result['hallucination_detected'] else 'No'}")
                  
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        return answer, from_kb, confidence
    
    # Replace the method
    chatbot_instance.get_answer_with_confidence = enhanced_get_answer_with_confidence
    return chatbot_instance

# Example usage and test cases
def create_sample_test_set():
    """Create sample test cases for evaluation."""
    return [
        {
            "question": "How do I prevent swarming in my beehive?",
            "answer": "To prevent swarming, ensure adequate space in the hive by adding supers before they're needed. Regular inspection every 7-14 days during swarming season is crucial. Remove queen cells if found early, and consider splitting the hive if the colony is very strong. Proper ventilation and avoiding overcrowding are key factors.",
            "contexts": [
                "Swarm prevention requires regular hive inspections during spring. Add honey supers before the hive becomes crowded. Remove queen cells during regular inspections to prevent swarming behavior.",
                "Overcrowding is the main cause of swarming. Provide adequate space and good ventilation. Strong colonies may need to be split to prevent swarming."
            ],
            "confidence": 0.85,
            "from_kb": True
        },
        {
            "question": "What's the best type of wood for beehive construction?",
            "answer": "Pine and cedar are excellent choices for beehive construction. Pine is cost-effective and easy to work with, while cedar naturally resists moisture and insects. Avoid treated lumber as chemicals can harm bees. The wood should be at least 3/4 inch thick for insulation.",
            "contexts": [
                "Pine wood is commonly used for beehive construction due to its availability and workability. Cedar is preferred for its natural resistance properties.",
                "Hive bodies should use untreated lumber that's at least 3/4 inch thick for proper insulation and durability."
            ],
            "confidence": 0.78,
            "from_kb": True
        }
    ]

if __name__ == "__main__":
    # Example usage
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("Please set GROQ_API_KEY in your environment variables")
        exit(1)
    
    # Initialize evaluator
    evaluator = RAGAnswerEvaluator(GROQ_API_KEY)
    
    # Run evaluation on sample data
    test_cases = create_sample_test_set()
    results_df = evaluator.evaluate_test_set(test_cases)
    
    # Generate and display report
    report = evaluator.generate_evaluation_report(results_df)
    print("\nEvaluation Report:")
    print("=" * 50)
    print(f"Total Evaluations: {report['summary']['total_evaluations']}")
    print(f"Average Faithfulness Score: {report['summary']['avg_faithfulness_score']:.2f}/5")
    print(f"Average Accuracy Score: {report['summary']['avg_accuracy_score']:.2f}/5")
    print(f"Hallucination Rate: {report['summary']['hallucination_rate']:.2%}")
    print(f"Average Semantic Similarity: {report['summary']['avg_semantic_similarity']:.3f}")
    print(f"Average Overall Quality: {report['summary']['avg_overall_quality']:.2f}/5")
    
    # Save results
    evaluator.save_evaluation_results("rag_evaluation_results.json", results_df)