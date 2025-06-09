from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from main_logic import app as langgraph_app

fastapi_app = FastAPI()

class Query(BaseModel):
    question: str
    thread_id: str

@fastapi_app.post("/ask")
async def ask_api(q: Query):
    try:
        cfg = {"configurable": {"thread_id": q.thread_id}}
        resp = langgraph_app.invoke(
            {"messages": [HumanMessage(content=q.question)]},
            config=cfg
        )
        return {
            "messages": [m.content for m in resp["messages"]],
            "done": resp.get("resolvido", False)
        }
    except Exception as e:                        # <- captura erro p/ não quebrar o JSON
        raise HTTPException(status_code=500, detail=str(e))
