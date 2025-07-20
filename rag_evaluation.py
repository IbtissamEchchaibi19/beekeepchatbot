import json
import re
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

warnings.filterwarnings("ignore")

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    exact_match: float
    f1_score: float
    semantic_similarity: float
    entailment_accuracy: float
    total_questions: int
    
    def __str__(self):
        return f"""
╔══════════════════════════════════════╗
║        RAG EVALUATION RESULTS        ║
╠══════════════════════════════════════╣
║ Total Questions: {self.total_questions:15} ║
║ Exact Match:     {self.exact_match:14.1%} ║
║ F1 Score:        {self.f1_score:14.1%} ║
║ Semantic Sim:    {self.semantic_similarity:14.1%} ║
║ Entailment Acc:  {self.entailment_accuracy:14.1%} ║
╚══════════════════════════════════════╝
        """

class RAGSystemEvaluator:
    """Comprehensive evaluation system for RAG models"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the evaluator
        
        Args:
            use_gpu: Whether to use GPU for sentence embeddings (if available)
        """
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu else 'cpu'
        
        # Initialize sentence transformer for semantic similarity
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.use_gpu:
            self.sentence_model = self.sentence_model.to(self.device)
        
        # Initialize NLI model for entailment (using sentence transformers)
        print("Loading NLI model for entailment...")
        try:
            self.nli_model = SentenceTransformer('all-MiniLM-L6-v2')  # We'll use similarity as proxy
            self.use_nli_proxy = True
        except Exception as e:
            print(f"Warning: Could not load advanced NLI model, using similarity proxy: {e}")
            self.use_nli_proxy = True
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        print("Evaluator initialized successfully!")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing punctuation, 
        converting to lowercase, and handling whitespace
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for meaningful ones in context
        # Keep periods that might be important for abbreviations
        text = re.sub(r'[^\w\s\.]', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def calculate_exact_match(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate exact match score after normalization
        
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_normalized = self.normalize_text(predicted)
        truth_normalized = self.normalize_text(ground_truth)
        
        return 1.0 if pred_normalized == truth_normalized else 0.0
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text and remove stopwords"""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        tokens = [token for token in tokens 
                 if token not in string.punctuation 
                 and token not in self.stop_words 
                 and token.strip()]
        
        return tokens
    
    def calculate_f1_score(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate F1 score based on token overlap
        
        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_tokens = set(self.tokenize_text(predicted))
        truth_tokens = set(self.tokenize_text(ground_truth))
        
        if len(truth_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Calculate precision and recall
        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(truth_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings
        
        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        if not predicted.strip() or not ground_truth.strip():
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([predicted, ground_truth])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Ensure the result is between 0 and 1
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_entailment_accuracy(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate entailment accuracy using NLI
        
        For now, we'll use a high semantic similarity threshold as a proxy
        In production, you might want to use a dedicated NLI model
        
        Returns:
            1.0 if entailed, 0.0 otherwise
        """
        if self.use_nli_proxy:
            # Use high semantic similarity as entailment proxy
            semantic_sim = self.calculate_semantic_similarity(predicted, ground_truth)
            return 1.0 if semantic_sim > 0.75 else 0.0
        
        # If you have a proper NLI model, implement it here
        # Example with transformers pipeline:
        # result = self.nli_pipeline(f"premise: {ground_truth} hypothesis: {predicted}")
        # return 1.0 if result['label'] == 'ENTAILMENT' else 0.0
        
        return 0.0
    
    def evaluate_single_pair(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {
            'exact_match': self.calculate_exact_match(predicted, ground_truth),
            'f1_score': self.calculate_f1_score(predicted, ground_truth),
            'semantic_similarity': self.calculate_semantic_similarity(predicted, ground_truth),
            'entailment_accuracy': self.calculate_entailment_accuracy(predicted, ground_truth)
        }
        
        return metrics
    
    def evaluate_dataset(self, predictions: List[str], ground_truths: List[str]) -> EvaluationMetrics:
        """
        Evaluate entire dataset
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            EvaluationMetrics object with aggregated scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
        
        all_metrics = []
        
        print(f"Evaluating {len(predictions)} question-answer pairs...")
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            if i % 10 == 0 and i > 0:
                print(f"Processed {i}/{len(predictions)} pairs...")
            
            metrics = self.evaluate_single_pair(pred, truth)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {
            'exact_match': np.mean([m['exact_match'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'semantic_similarity': np.mean([m['semantic_similarity'] for m in all_metrics]),
            'entailment_accuracy': np.mean([m['entailment_accuracy'] for m in all_metrics])
        }
        
        return EvaluationMetrics(
            exact_match=aggregated['exact_match'],
            f1_score=aggregated['f1_score'],
            semantic_similarity=aggregated['semantic_similarity'],
            entailment_accuracy=aggregated['entailment_accuracy'],
            total_questions=len(predictions)
        )
    
    def evaluate_from_jsonl(self, jsonl_file_path: str, rag_system, 
                           session_id: str = "eval_session") -> EvaluationMetrics:
        """
        Evaluate RAG system using JSONL ground truth file
        
        Args:
            jsonl_file_path: Path to JSONL file with ground truth
            rag_system: Your RAG system instance
            session_id: Session ID for RAG system
            
        Returns:
            EvaluationMetrics object
        """
        print(f"Loading ground truth from {jsonl_file_path}")
        
        # Load ground truth data
        ground_truth_data = []
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            ground_truth_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {jsonl_file_path}")
        
        if not ground_truth_data:
            raise ValueError("No valid data found in ground truth file")
        
        print(f"Loaded {len(ground_truth_data)} ground truth examples")
        
        # Generate predictions
        predictions = []
        ground_truths = []
        
        print("Generating predictions from RAG system...")
        
        for i, item in enumerate(ground_truth_data):
            if i % 5 == 0 and i > 0:
                print(f"Generated {i}/{len(ground_truth_data)} predictions...")
            
            question = item.get('instruction', '')
            ground_truth = item.get('output', '')
            
            if not question or not ground_truth:
                print(f"Warning: Skipping item {i} due to missing question or answer")
                continue
            
            # Get prediction from RAG system
            try:
                if hasattr(rag_system, 'get_answer_with_confidence'):
                    prediction, _, _ = rag_system.get_answer_with_confidence(question, session_id)
                elif hasattr(rag_system, 'chat_interface'):
                    history = []
                    _, history = rag_system.chat_interface(question, history, session_id)
                    prediction = history[-1]['content'] if history else ""
                else:
                    print("Warning: RAG system doesn't have expected methods")
                    prediction = "No answer generated"
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
            except Exception as e:
                print(f"Error generating prediction for item {i}: {e}")
                predictions.append("")
                ground_truths.append(ground_truth)
        
        print(f"Generated {len(predictions)} predictions")
        
        # Evaluate
        return self.evaluate_dataset(predictions, ground_truths)
    
    def detailed_evaluation_report(self, predictions: List[str], ground_truths: List[str], 
                                 questions: Optional[List[str]] = None) -> str:
        """
        Generate detailed evaluation report with per-question breakdown
        
        Returns:
            Formatted string report
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Mismatched lengths")
        
        all_metrics = []
        for pred, truth in zip(predictions, ground_truths):
            metrics = self.evaluate_single_pair(pred, truth)
            all_metrics.append(metrics)
        
        # Overall metrics
        overall = self.evaluate_dataset(predictions, ground_truths)
        
        # Find best and worst performing examples
        f1_scores = [m['f1_score'] for m in all_metrics]
        best_idx = np.argmax(f1_scores)
        worst_idx = np.argmin(f1_scores)
        
        report = f"""
{overall}

📊 DETAILED ANALYSIS:
====================

Best Performing Example (F1: {f1_scores[best_idx]:.3f}):
Question: {questions[best_idx] if questions else f"Example {best_idx}"}
Predicted: {predictions[best_idx][:100]}...
Ground Truth: {ground_truths[best_idx][:100]}...

Worst Performing Example (F1: {f1_scores[worst_idx]:.3f}):
Question: {questions[worst_idx] if questions else f"Example {worst_idx}"}
Predicted: {predictions[worst_idx][:100]}...
Ground Truth: {ground_truths[worst_idx][:100]}...

📈 SCORE DISTRIBUTION:
- Exact Match: {np.sum([m['exact_match'] for m in all_metrics])} / {len(all_metrics)} pairs
- High F1 (>0.8): {np.sum([1 for f1 in f1_scores if f1 > 0.8])} / {len(all_metrics)} pairs
- High Semantic Sim (>0.8): {np.sum([1 for m in all_metrics if m['semantic_similarity'] > 0.8])} / {len(all_metrics)} pairs
        """
        
        return report

# Example usage and integration functions

def evaluate_rag_system(rag_system, ground_truth_jsonl: str, 
                       session_id: str = "evaluation") -> EvaluationMetrics:
    """
    Main function to evaluate a RAG system
    
    Args:
        rag_system: Your RAG system instance
        ground_truth_jsonl: Path to ground truth JSONL file
        session_id: Session ID for evaluation
        
    Returns:
        EvaluationMetrics object
    """
    evaluator = RAGSystemEvaluator(use_gpu=False)  # Set to True if you have GPU
    
    try:
        metrics = evaluator.evaluate_from_jsonl(
            ground_truth_jsonl, 
            rag_system, 
            session_id
        )
        
        print(metrics)
        return metrics
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

def evaluate_predictions_directly(predictions: List[str], ground_truths: List[str], 
                                questions: Optional[List[str]] = None) -> EvaluationMetrics:
    """
    Evaluate predictions directly without RAG system
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers  
        questions: Optional list of questions for detailed report
        
    Returns:
        EvaluationMetrics object
    """
    evaluator = RAGSystemEvaluator()
    metrics = evaluator.evaluate_dataset(predictions, ground_truths)
    
    print(metrics)
    
    if questions:
        detailed_report = evaluator.detailed_evaluation_report(
            predictions, ground_truths, questions
        )
        print(detailed_report)
    
    return metrics

# Integration example for your existing code
def add_evaluation_to_existing_rag():
    """
    Example of how to integrate evaluation into your existing RAG pipeline
    """
    
    # Add this method to your HoneyExpertChatbot class:
    """
    def evaluate_performance(self, ground_truth_jsonl: str, 
                           session_id: str = "eval") -> EvaluationMetrics:
        '''Evaluate this RAG system using ground truth data'''
        return evaluate_rag_system(self, ground_truth_jsonl, session_id)
    """
    
    # Then use it like this:
    """
    # In your main function or wherever you want to run evaluation
    chatbot = initialize_chatbot()
    setup_chatbot_efficiently(chatbot, "beekeeping_data.jsonl")
    
    # Run evaluation
    evaluation_metrics = chatbot.evaluate_performance("ground_truth.jsonl")
    print(f"System Performance: {evaluation_metrics}")
    """

if __name__ == "__main__":
    # Example usage
    print("RAG System Evaluator")
    print("===================")
    
    # Example with dummy data
    sample_predictions = [
        "Hybrid queens combine traits from local and European bees for better performance.",
        "Regular hive inspections should be done weekly during active season.",
        "Honey should be stored in airtight containers in cool, dry places."
    ]
    
    sample_ground_truths = [
        "Hybrid queens, bred from local and European stock, aim to combine traits like heat tolerance and high productivity, offering balanced performance in varied climatic conditions.",
        "Regular hive inspections are essential and should be conducted weekly during the active beekeeping season.",
        "Proper honey storage requires airtight containers in cool, dry environments to maintain quality."
    ]
    
    sample_questions = [
        "What role do hybrid queens play in modern beekeeping?",
        "How often should hive inspections be conducted?", 
        "What are the best practices for honey storage?"
    ]
    
    # Run evaluation
    print("Running sample evaluation...")
    evaluate_predictions_directly(sample_predictions, sample_ground_truths, sample_questions)