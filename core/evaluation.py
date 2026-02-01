"""
RAG Evaluation Module - MRR (Mean Reciprocal Rank) Accuracy

This module evaluates the retrieval quality of the RAG system using MRR,
which measures how well the system ranks relevant documents.

MRR Formula:
    MRR = (1/|Q|) * Σ (1/rank_i)
    
Where rank_i is the position of the first relevant document for query i.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Setup paths
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
EVAL_DATA_PATH = str(Path(__file__).parent.parent / "evaluation_data.json")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single query"""
    query: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    reciprocal_rank: float
    first_relevant_rank: Optional[int] = None
    is_hit: bool = False


@dataclass
class MRREvaluationReport:
    """Complete MRR evaluation report"""
    mrr_score: float
    total_queries: int
    hits: int
    misses: int
    hit_rate: float
    results: list[EvaluationResult] = field(default_factory=list)
    
    def __str__(self):
        report = f"""
{'='*60}
            RAG EVALUATION REPORT - MRR ACCURACY
{'='*60}

📊 SUMMARY METRICS
{'─'*40}
  Mean Reciprocal Rank (MRR): {self.mrr_score:.4f}
  Hit Rate:                   {self.hit_rate:.2%}
  Total Queries:              {self.total_queries}
  Hits (relevant found):      {self.hits}
  Misses (not found):         {self.misses}

📋 DETAILED RESULTS
{'─'*40}
"""
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.is_hit else "❌"
            rank_info = f"Rank: {result.first_relevant_rank}" if result.first_relevant_rank else "Not found"
            report += f"""
  {i}. {status} Query: "{result.query[:50]}{'...' if len(result.query) > 50 else ''}"
     Expected: {result.expected_sources}
     Retrieved: {result.retrieved_sources[:3]}{'...' if len(result.retrieved_sources) > 3 else ''}
     {rank_info} | RR: {result.reciprocal_rank:.4f}
"""
        
        report += f"""
{'='*60}
  MRR Score Interpretation:
  • 1.0 = Perfect (relevant doc always first)
  • 0.5 = Relevant doc at position 2 on average
  • 0.33 = Relevant doc at position 3 on average
  • 0.0 = No relevant documents found
{'='*60}
"""
        return report


def get_default_test_data() -> list[dict]:
    """
    Returns default test queries with expected relevant sources.
    Customize this based on your knowledge base.
    """
    return [
        {
            "query": "What programs does Orchid International College offer?",
            "relevant_sources": ["02_BSc_CSIT.md", "03_BCA.md", "04_BITM.md", "05_BBM.md", "06_BBS.md", "07_BSW.md"]
        },
        {
            "query": "Tell me about BSc CSIT program",
            "relevant_sources": ["02_BSc_CSIT.md"]
        },
        {
            "query": "What is the BCA program about?",
            "relevant_sources": ["03_BCA.md"]
        },
        {
            "query": "Information about BITM course",
            "relevant_sources": ["04_BITM.md"]
        },
        {
            "query": "What is BBM program?",
            "relevant_sources": ["05_BBM.md"]
        },
        {
            "query": "Tell me about BBS course",
            "relevant_sources": ["06_BBS.md"]
        },
        {
            "query": "What is BSW program?",
            "relevant_sources": ["07_BSW.md"]
        },
        {
            "query": "How can I contact Orchid International College?",
            "relevant_sources": ["08_Contact.md"]
        },
        {
            "query": "What is the history of Orchid International College?",
            "relevant_sources": ["01_About_Us.md"]
        },
        {
            "query": "Where is the college located?",
            "relevant_sources": ["08_Contact.md", "01_About_Us.md"]
        },
        {
            "query": "What are the admission requirements?",
            "relevant_sources": ["02_BSc_CSIT.md", "03_BCA.md", "04_BITM.md", "05_BBM.md", "06_BBS.md", "07_BSW.md"]
        },
        {
            "query": "Tell me about the faculty",
            "relevant_sources": ["01_About_Us.md"]
        },
        {
            "query": "What is the fee structure?",
            "relevant_sources": ["02_BSc_CSIT.md", "03_BCA.md", "04_BITM.md", "05_BBM.md", "06_BBS.md", "07_BSW.md"]
        },
        {
            "query": "Duration of BSc CSIT program",
            "relevant_sources": ["02_BSc_CSIT.md"]
        },
        {
            "query": "Career opportunities after BCA",
            "relevant_sources": ["03_BCA.md"]
        }
    ]


def load_test_data(filepath: Optional[str] = None) -> list[dict]:
    """Load test data from JSON file or return default data"""
    if filepath and Path(filepath).exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return get_default_test_data()


def save_test_data(test_data: list[dict], filepath: str = EVAL_DATA_PATH):
    """Save test data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Test data saved to {filepath}")


def calculate_reciprocal_rank(
    retrieved_docs: list[Document], 
    relevant_sources: list[str]
) -> tuple[float, Optional[int]]:
    """
    Calculate Reciprocal Rank for a single query.
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_sources: List of expected relevant source filenames
        
    Returns:
        Tuple of (reciprocal_rank, first_relevant_position)
    """
    for rank, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', '')
        # Extract just the filename from the full path
        source_filename = Path(source).name
        
        if source_filename in relevant_sources or any(
            rel_src in source for rel_src in relevant_sources
        ):
            return 1.0 / rank, rank
    
    return 0.0, None


def evaluate_mrr(
    test_data: Optional[list[dict]] = None,
    k: int = 10,
    verbose: bool = True
) -> MRREvaluationReport:
    """
    Evaluate the RAG retrieval system using Mean Reciprocal Rank.
    
    Args:
        test_data: List of test queries with expected relevant sources.
                   Each item should have 'query' and 'relevant_sources' keys.
        k: Number of documents to retrieve for each query
        verbose: Whether to print progress
        
    Returns:
        MRREvaluationReport with detailed results
    """
    if test_data is None:
        test_data = get_default_test_data()
    
    # Initialize vector store
    vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    results = []
    total_rr = 0.0
    hits = 0
    
    if verbose:
        print(f"\n🔍 Evaluating {len(test_data)} queries with k={k}...\n")
    
    for i, item in enumerate(test_data):
        query = item['query']
        relevant_sources = item['relevant_sources']
        
        # Retrieve documents
        retrieved_docs = retriever.invoke(query)
        retrieved_sources = [
            Path(doc.metadata.get('source', '')).name 
            for doc in retrieved_docs
        ]
        
        # Calculate reciprocal rank
        rr, first_rank = calculate_reciprocal_rank(retrieved_docs, relevant_sources)
        total_rr += rr
        
        is_hit = rr > 0
        if is_hit:
            hits += 1
        
        result = EvaluationResult(
            query=query,
            expected_sources=relevant_sources,
            retrieved_sources=retrieved_sources,
            reciprocal_rank=rr,
            first_relevant_rank=first_rank,
            is_hit=is_hit
        )
        results.append(result)
        
        if verbose:
            status = "✅" if is_hit else "❌"
            print(f"  {status} Query {i+1}/{len(test_data)}: RR={rr:.4f}")
    
    # Calculate MRR
    mrr = total_rr / len(test_data) if test_data else 0.0
    misses = len(test_data) - hits
    hit_rate = hits / len(test_data) if test_data else 0.0
    
    report = MRREvaluationReport(
        mrr_score=mrr,
        total_queries=len(test_data),
        hits=hits,
        misses=misses,
        hit_rate=hit_rate,
        results=results
    )
    
    return report


def evaluate_mrr_at_k(
    test_data: Optional[list[dict]] = None,
    k_values: list[int] = [1, 3, 5, 10]
) -> dict[int, float]:
    """
    Evaluate MRR at different values of k.
    
    Args:
        test_data: Test queries with expected sources
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary mapping k to MRR score
    """
    if test_data is None:
        test_data = get_default_test_data()
    
    results = {}
    print(f"\n📊 Evaluating MRR@k for k in {k_values}...\n")
    
    for k in k_values:
        report = evaluate_mrr(test_data, k=k, verbose=False)
        results[k] = report.mrr_score
        print(f"  MRR@{k}: {report.mrr_score:.4f} (Hit Rate: {report.hit_rate:.2%})")
    
    return results


def run_full_evaluation(save_results: bool = True) -> MRREvaluationReport:
    """
    Run a comprehensive MRR evaluation and optionally save results.
    
    Args:
        save_results: Whether to save results to a JSON file
        
    Returns:
        Complete evaluation report
    """
    print("\n" + "="*60)
    print("     Starting RAG MRR Evaluation")
    print("="*60)
    
    # Load or use default test data
    test_data = load_test_data(EVAL_DATA_PATH)
    
    # Run main evaluation
    report = evaluate_mrr(test_data, k=10, verbose=True)
    
    # Print report
    print(report)
    
    # Run MRR@k analysis
    print("\n📈 MRR@k Analysis:")
    mrr_at_k = evaluate_mrr_at_k(test_data, k_values=[1, 3, 5, 10])
    
    # Save results
    if save_results:
        results_path = Path(__file__).parent.parent / "evaluation_results.json"
        results_data = {
            "mrr_score": report.mrr_score,
            "hit_rate": report.hit_rate,
            "total_queries": report.total_queries,
            "hits": report.hits,
            "misses": report.misses,
            "mrr_at_k": mrr_at_k,
            "detailed_results": [
                {
                    "query": r.query,
                    "expected_sources": r.expected_sources,
                    "retrieved_sources": r.retrieved_sources,
                    "reciprocal_rank": r.reciprocal_rank,
                    "first_relevant_rank": r.first_relevant_rank,
                    "is_hit": r.is_hit
                }
                for r in report.results
            ]
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {results_path}")
    
    return report


if __name__ == "__main__":
    report = run_full_evaluation(save_results=True)
