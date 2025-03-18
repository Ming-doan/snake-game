import torch
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from model import Linear_QNet


class State(BaseModel):
    is_danger_straight: bool
    is_danger_right: bool
    is_danger_left: bool
    move_left: bool
    move_right: bool
    move_up: bool
    move_down: bool
    food_left: bool
    food_right: bool
    food_up: bool
    food_down: bool


app = FastAPI(docs_url="/")
MODEL = Linear_QNet(11, 256, 3)
MODEL.load_state_dict(torch.load("./model/model_600_epochs.pth"))
MODEL.eval()
MOVE_OPTIONS = ["none", "right", "left"]


async def _handler(state: State) -> str:
    print("🐍", state)
    torch_state = torch.tensor([
        state.is_danger_straight,
        state.is_danger_right,
        state.is_danger_left,
        state.move_left,
        state.move_right,
        state.move_up,
        state.move_down,
        state.food_left,
        state.food_right,
        state.food_up,
        state.food_down
    ], dtype=torch.float)
    prediction = MODEL(torch_state)
    move_idx = torch.argmax(prediction).item()
    return MOVE_OPTIONS[move_idx]


@app.websocket("/move")
async def move_snake(websocket: WebSocket, state: State):
    move = await _handler(state)
    await websocket.send_text(move)


@app.post("/move")
async def move_snake(state: State):
    return await _handler(state)
