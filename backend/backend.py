import os
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from interpreter import basic

app = FastAPI()

# Comma-separated list, e.g. "http://localhost:4000,https://d123.cloudfront.net"
cors_raw = os.environ.get(
    "CORS_ORIGINS", "http://localhost:4000"
)
cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class InterpreterType(BaseModel):
    type: str
    value: str | int | float

    def from_json(self):
        if self.type == "number":
            return basic.Number(value=self.value)
        elif self.type == "function":
            # TODO: this is probably wrong :(
            return basic.Function(name=self.name, body_node=None, arg_names=self.arg_names)
        else:
            raise NotImplementedError(f'missing implementation for {self.type}')

    def to_json(self):
        return {"type": type(self).__name__.lower(), "value": repr(self)}


class InterpretRequest(BaseModel):
    code: str
    symbols: Optional[Dict[str, InterpreterType]] = None


@app.post("/interpret")
def interpret(req: InterpretRequest)-> Dict:
    symbol_table = basic.SymbolTable.from_json(req.symbols)
    res, context = basic.run_ai("<stdin>", req.code, symbol_table=symbol_table, force_ai=False)
    if res.error:
        return {"result": None, "symbol_table": None, "error": res.error.as_string()}

    return {
        "result": str(res.value) if res.value is not None else "",
        "symbols": context.symbol_table.to_json(),
        "error": None,
    }
