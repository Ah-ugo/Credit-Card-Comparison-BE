from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import os
from pymongo import MongoClient
from bson import ObjectId
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Comparison API",
    description="AI-powered credit card comparison platform for Indian banks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    model = None

# MongoDB connection
try:
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    db = client.credit_cards
    cards_collection = db.cards
    queries_collection = db.user_queries
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    db = None

# Pydantic models
class CreditCard(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    name: str
    bank: str
    network: str = "Visa"
    type: str = "Premium"
    annual_fee: int = 0
    joining_fee: int = 0
    add_on_card_fee: int = 0
    foreign_transaction_fee: float = 3.5
    interest_rate: float = 3.49
    cash_advance_fee: float = 2.5
    late_payment_fee: int = 750
    over_limit_fee: int = 600
    rewards_rate: float = 1.0
    cashback_rate: float = 0.0
    min_income: int = 600000
    min_credit_score: int = 700
    age_requirement: str = "21-60 years"
    documents_required: List[str] = ["PAN Card", "Aadhaar Card", "Income Proof"]
    summary: str
    ai_summary: Optional[str] = None
    tags: List[str] = []
    benefits: List[str] = []
    rewards_structure: List[str] = []
    welcome_offers: List[str] = []
    fee_waiver_conditions: List[str] = []
    rating: float = 4.0
    reviews: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}

class CreditCardCreate(BaseModel):
    name: str
    bank: str
    network: str = "Visa"
    type: str = "Premium"
    annual_fee: int = 0
    joining_fee: int = 0
    add_on_card_fee: int = 0
    foreign_transaction_fee: float = 3.5
    interest_rate: float = 3.49
    cash_advance_fee: float = 2.5
    late_payment_fee: int = 750
    over_limit_fee: int = 600
    rewards_rate: float = 1.0
    cashback_rate: float = 0.0
    min_income: int = 600000
    min_credit_score: int = 700
    age_requirement: str = "21-60 years"
    documents_required: List[str] = ["PAN Card", "Aadhaar Card", "Income Proof"]
    summary: str
    tags: List[str] = []
    benefits: List[str] = []
    rewards_structure: List[str] = []
    welcome_offers: List[str] = []
    fee_waiver_conditions: List[str] = []
    rating: float = 4.0
    reviews: int = 0


class CreditCardUpdate(BaseModel):
    name: Optional[str] = None
    bank: Optional[str] = None
    network: Optional[str] = None
    type: Optional[str] = None
    annual_fee: Optional[int] = None
    joining_fee: Optional[int] = None
    add_on_card_fee: Optional[int] = None
    foreign_transaction_fee: Optional[float] = None
    interest_rate: Optional[float] = None
    cash_advance_fee: Optional[float] = None
    late_payment_fee: Optional[int] = None
    over_limit_fee: Optional[int] = None
    rewards_rate: Optional[float] = None
    cashback_rate: Optional[float] = None
    min_income: Optional[int] = None
    min_credit_score: Optional[int] = None
    age_requirement: Optional[str] = None
    documents_required: Optional[List[str]] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    benefits: Optional[List[str]] = None
    rewards_structure: Optional[List[str]] = None
    welcome_offers: Optional[List[str]] = None
    fee_waiver_conditions: Optional[List[str]] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None

class SearchRequest(BaseModel):
    query: str

class CompareRequest(BaseModel):
    card1_id: str
    card2_id: str

class SearchResponse(BaseModel):
    cards: List[Dict[str, Any]]
    ai_response: str

class CompareResponse(BaseModel):
    card1: Dict[str, Any]
    card2: Dict[str, Any]
    ai_summary: str

# Helper functions
def serialize_card(card):
    """Convert MongoDB document to JSON serializable format"""
    if card is None:
        return None
    if "_id" in card:
        card["id"] = str(card["_id"])
        del card["_id"]
    return card

def generate_ai_summary(card_data):
    """Generate AI summary for a credit card"""
    if not model:
        return "AI summary not available"
    try:
        prompt = f"""
        Generate a comprehensive summary for this credit card that highlights its key features, benefits, and ideal user profile:

        Card: {card_data.get('name')} ({card_data.get('bank')})
        Type: {card_data.get('type')}
        Annual Fee: ₹{card_data.get('annual_fee', 0)}
        Rewards Rate: {card_data.get('rewards_rate', 1)}x
        Benefits: {', '.join(card_data.get('benefits', []))}
        Tags: {', '.join(card_data.get('tags', []))}

        Provide a detailed summary (2-3 sentences) that helps users understand if this card is right for them.
        Focus on the key value propositions and target audience.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate AI summary: {e}")
        return "AI summary not available"

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Credit Card Comparison API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mongodb": "connected" if db else "disconnected",
        "gemini_ai": "configured" if model else "not configured"
    }

@app.get("/cards")
async def get_all_cards():
    """Get all credit cards"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        cards = list(cards_collection.find({}))
        serialized_cards = [serialize_card(card) for card in cards]
        logger.info(f"Retrieved {len(serialized_cards)} cards")
        return serialized_cards
    except Exception as e:
        logger.error(f"Error fetching cards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cards/{card_id}")
async def get_card(card_id: str):
    """Get a specific credit card by ID"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        card = cards_collection.find_one({"_id": ObjectId(card_id)})
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")
        return serialize_card(card)
    except Exception as e:
        logger.error(f"Error fetching card {card_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cards")
async def create_card(card_data: CreditCardCreate):
    """Create a new credit card"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        card_dict = card_data.dict()
        card_dict["created_at"] = datetime.utcnow()
        card_dict["updated_at"] = datetime.utcnow()
        card_dict["ai_summary"] = generate_ai_summary(card_dict)
        result = cards_collection.insert_one(card_dict)
        created_card = cards_collection.find_one({"_id": result.inserted_id})
        logger.info(f"Created new card: {card_data.name}")
        return serialize_card(created_card)
    except Exception as e:
        logger.error(f"Error creating card: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/cards/{card_id}")
async def update_card(card_id: str, card_data: CreditCardUpdate):
    """Update an existing credit card"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        card_dict = card_data.dict()
        card_dict["updated_at"] = datetime.utcnow()
        card_dict["ai_summary"] = generate_ai_summary(card_dict)
        result = cards_collection.update_one(
            {"_id": ObjectId(card_id)},
            {"$set": card_dict}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Card not found")
        updated_card = cards_collection.find_one({"_id": ObjectId(card_id)})
        logger.info(f"Updated card: {card_id}")
        return serialize_card(updated_card)
    except Exception as e:
        logger.error(f"Error updating card {card_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cards/{card_id}")
async def delete_card(card_id: str):
    """Delete a credit card"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        result = cards_collection.delete_one({"_id": ObjectId(card_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Card not found")
        logger.info(f"Deleted card: {card_id}")
        return {"message": "Card deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting card {card_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_cards(request: SearchRequest):
    """Search credit cards using natural language query with Gemini AI"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        query_doc = {
            "query": request.query,
            "timestamp": datetime.utcnow()
        }
        queries_collection.insert_one(query_doc)
        all_cards = list(cards_collection.find({}))
        serialized_cards = [serialize_card(card) for card in all_cards]
        ai_response = "No AI response available"
        if model and serialized_cards:
            try:
                cards_info = "\n".join([
                    f"Card: {card['name']} ({card['bank']}) - Annual Fee: ₹{card.get('annual_fee', 0)} - "
                    f"Rewards: {card.get('rewards_rate', 1)}x - Summary: {card.get('summary', '')} - "
                    f"Tags: {', '.join(card.get('tags', []))}"
                    for card in serialized_cards
                ])
                prompt = f"""
                You are a credit card recommendation expert for Indian banks. 
                Based on the following query, provide a helpful response that recommends specific credit cards from the available options:

                Query: "{request.query}"

                Available Cards:
                {cards_info}

                Provide a concise, helpful response (2-3 sentences) that directly addresses the query with specific card recommendations.
                Focus on being informative and precise. Mention specific card names and why they match the user's needs.
                """
                response = model.generate_content(prompt)
                ai_response = response.text
            except Exception as e:
                logger.error(f"Error generating AI response: {e}")
                ai_response = "I understand your query, but I'm having trouble generating a detailed response right now. Please browse the available cards below."
        query_lower = request.query.lower()
        keywords = query_lower.split()
        filtered_cards = []
        for card in serialized_cards:
            if any(
                keyword in card.get('name', '').lower() or
                keyword in card.get('summary', '').lower() or
                keyword in card.get('bank', '').lower() or
                any(keyword in tag.lower() for tag in card.get('tags', [])) or
                any(keyword in benefit.lower() for benefit in card.get('benefits', []))
                for keyword in keywords
            ):
                filtered_cards.append(card)
        if not filtered_cards:
            filtered_cards = serialized_cards[:6]
        logger.info(f"Search query: '{request.query}' returned {len(filtered_cards)} cards")
        return {
            "cards": filtered_cards,
            "ai_response": ai_response
        }
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_cards(request: CompareRequest):
    """Compare two credit cards with AI-generated summary"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        card1 = cards_collection.find_one({"_id": ObjectId(request.card1_id)})
        card2 = cards_collection.find_one({"_id": ObjectId(request.card2_id)})
        if not card1 or not card2:
            raise HTTPException(status_code=404, detail="One or both cards not found")
        card1_serialized = serialize_card(card1)
        card2_serialized = serialize_card(card2)
        ai_summary = "Comparison not available"
        if model:
            try:
                prompt = f"""
                Compare these two credit cards and provide a concise summary of their key differences and which might be better for different types of users:

                Card 1: {card1_serialized['name']} ({card1_serialized['bank']})
                Annual Fee: ₹{card1_serialized.get('annual_fee', 0)}
                Rewards Rate: {card1_serialized.get('rewards_rate', 1)}x
                Key Benefits: {', '.join(card1_serialized.get('benefits', []))}

                Card 2: {card2_serialized['name']} ({card2_serialized['bank']})
                Annual Fee: ₹{card2_serialized.get('annual_fee', 0)}
                Rewards Rate: {card2_serialized.get('rewards_rate', 1)}x
                Key Benefits: {', '.join(card2_serialized.get('benefits', []))}

                Provide a concise, helpful comparison (3-4 sentences) focusing on the key differences and who each card might be better for.
                """
                response = model.generate_content(prompt)
                ai_summary = response.text
            except Exception as e:
                logger.error(f"Error generating comparison summary: {e}")
                ai_summary = "Both cards have their unique benefits. Please review the detailed comparison above to make an informed decision."
        logger.info(f"Compared cards: {request.card1_id} vs {request.card2_id}")
        return {
            "card1": card1_serialized,
            "card2": card2_serialized,
            "ai_summary": ai_summary
        }
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cards/{card_id}/generate-summary")
async def generate_card_summary(card_id: str):
    """Generate AI summary for a specific card"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        card = cards_collection.find_one({"_id": ObjectId(card_id)})
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")
        ai_summary = generate_ai_summary(card)
        cards_collection.update_one(
            {"_id": ObjectId(card_id)},
            {"$set": {"ai_summary": ai_summary, "updated_at": datetime.utcnow()}}
        )
        logger.info(f"Generated AI summary for card: {card_id}")
        return {"ai_summary": ai_summary}
    except Exception as e:
        logger.error(f"Error generating summary for card {card_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/queries")
async def get_query_analytics():
    """Get analytics on user queries"""
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Database not available")
        recent_queries = list(queries_collection.find({}).sort("timestamp", -1).limit(100))
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": -1}},
            {"$limit": 30}
        ]
        daily_counts = list(queries_collection.aggregate(pipeline))
        return {
            "recent_queries": [serialize_card(q) for q in recent_queries],
            "daily_counts": daily_counts,
            "total_queries": queries_collection.count_documents({})
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)