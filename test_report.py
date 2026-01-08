from services import report_service
import asyncio

async def test():
    print("Testing Technical Aggregator...")
    tech = await report_service.get_master_score("SOLUSDT")
    print(f"Technical Score: {tech}")
    
    print("\nTesting Sentiment Combo...")
    sent = report_service.get_sentiment_ma(days=7)
    print(f"7-Day Sentiment MA: {sent}")

if __name__ == "__main__":
    asyncio.run(test())
