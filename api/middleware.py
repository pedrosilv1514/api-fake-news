"""Bound request bytes before JSON parsing; endpoint contracts add semantic limits."""
import asyncio

from starlette.responses import JSONResponse


class BoundedBodyMiddleware:
    def __init__(self, app, max_bytes=512_000):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"]!="http" or scope["method"] not in {"POST","PUT","PATCH"}:
            return await self.app(scope,receive,send)
        chunks, total = [], 0
        loop=asyncio.get_running_loop()
        deadline=loop.time()+5
        while True:
            try: message=await asyncio.wait_for(receive(),timeout=max(.001,deadline-loop.time()))
            except TimeoutError:
                return await JSONResponse({"detail":"Prazo de envio do corpo excedido"},status_code=408)(scope,receive,send)
            if message["type"]=="http.disconnect": return
            chunk=message.get("body",b"")
            total+=len(chunk)
            if total>self.max_bytes:
                return await JSONResponse({"detail":"Corpo excede 512000 bytes"},status_code=413)(scope,receive,send)
            chunks.append(chunk)
            if not message.get("more_body",False): break
        consumed=False
        async def replay():
            nonlocal consumed
            if not consumed:
                consumed=True
                return {"type":"http.request","body":b"".join(chunks),"more_body":False}
            return await receive()
        await self.app(scope,replay,send)

