import os
from types import SimpleNamespace

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, Field

from llm.azure_openai import get_structured_completion
from vectorstore.azure_ai_search import AzureAISearchVectorStore

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | None = None
    net_income: str | None = None
    operating_income: str | None = None
    cash_flow: str | None = None
    total_assets: str | None = None
    total_liabilities: str | None = None
    risk_factors: list[str] = []
    growth_drivers: list[str] = []


class Retriever:
    def __init__(self, client):
        self.client = client

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """
        Retrieve relevant chunks from Azure AI Search.
        """
        filter_expr = None

        if company and year:
            filter_expr = (
                f"company eq '{company}' "
                f"and year eq '{year}'"
            )

        results = (
            self.client.search(
                search_text=query,
                top=top_k,
                filter=filter_expr
            )
            if filter_expr
            else self.client.search(
                search_text=query,
                top=top_k
            )
        )

        documents = []

        for result in results:
            content = result.get("content", "")
            documents.append(
                SimpleNamespace(
                    page_content=content
                )
            )

        return documents


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int
) -> str:
    """
    Retrieve broad financial context from the vector store.
    """
    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year}
    """

    documents = retriever.invoke(
        query=query,
        company=company,
        year=year,
        top_k=20
    )
    # print(documents)
    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null for any value not found in the context.
- revenue, net_income, operating_income, cash_flow, total_assets and
  total_liabilities must each be ONE single value for the fiscal year,
  written exactly as reported, including units (e.g. "$391,035 million").
  Never return breakdowns or nested objects.
- revenue is total net sales only.
- risk_factors: list 6 to 8 of the most material risks. Each item should be a
  full descriptive sentence that includes the specific cause and any figures,
  percentages or named regulations/cases mentioned in the context.
- growth_drivers: list 5 to 7 key drivers. Each item should be a full
  descriptive sentence that includes specific numbers from the context
  (growth %, margin %, named products or services, regions, and comparisons
  with the prior year).
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using RAG.
    """
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    metrics = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )

    return metrics.model_dump()


def main() -> None:
    company = "Apple"
    year = 2024

    vector_store = AzureAISearchVectorStore(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        api_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
    )

    retriever = Retriever(
        vector_store.client
    )

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)


    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__ == "__main__":
    main()