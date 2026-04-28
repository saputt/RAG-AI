import redis
import json

class RedisMemory:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def add_message(self, role, content, roomid):
        key = f"chat:{roomid}"
        message = json.dumps({"role" : role, "content" : content})

        self.client.rpush(key, message)

        self.client.ltrim(key, -20, -1)

        self.client.expire(key, 172800)

    def get_messages(self, roomid):
        key=f"chat:{roomid}"

        raw_messages = self.client.lrange(key, 0, -1)

        return [json.loads(m) for m in raw_messages]

    def delete_message(self, roomid):
        key = f"chat:{roomid}"

        self.client.delete(key)
