import requests


class askanythingLLM:
    def __init__(self, slug="attack", api_key="1EXF0TW-QNH4AP7-N5Q7Z2C-HP50DAV"):
        self.slug = slug
        self.api_key = api_key

    def query(self, question):
        # url=f"http://10.201.101.207:3001/api/v1/workspace/{slug}/chat"
        # 192.168.192.237
        slug = self.slug
        url = f"http://192.168.192.237:3001/api/v1/workspace/{slug}/chat"
        api_key = self.api_key

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        data = {"message": question, "mode": "chat"}  # chat或者query
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            # 去除思考过程
            answer = result["textResponse"].split("</think>")[-1].strip()  # ？？？
            # answer = result["textResponse"].strip()  # ？？？
            sources = result.get("sources", [])
            # return answer, sources
            return answer
        else:
            return f"Error {response.text}", "jhf"


if __name__ == "__main__":
    api_key = "1EXF0TW-QNH4AP7-N5Q7Z2C-HP50DAV"
    slug = "try"
    question = "那么王世恒干了什么?"
    # answer, sources = askanythingLLM(question, slug, api_key)
    response = askanythingLLM(question, slug, api_key)
    # print("回答:", answer)
    # print("来源:", [src["title"] for src in sources])
    print(response)
