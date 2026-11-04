import requests

url = "http://127.0.0.1:8000/ask"

CONNECT_TIMEOUT = 5    
READ_TIMEOUT = 90     

print("--- RAG System Continuous Chat Client ---")
print("Type 'exit', 'quit', or 'bye' to end the chat.\n")

while True:
    try:
        user_query = input("\nYou: ").strip()

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("\nChat ended. Goodbye!")
            break

        if not user_query:
            print("Question cannot be empty. Please try again.")
            continue

        payload = {"query_str": user_query}

        print("\nResponse: ", end="", flush=True)

        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)

        print("\n" + "-" * 50)

    except requests.exceptions.ConnectTimeout:
        print("\nCould not connect to the server (timeout). Check that it is running.")
    except requests.exceptions.ReadTimeout:
        print("\nThe server took too long to respond (timeout). Please try again.")
    except requests.exceptions.HTTPError as e:
        print(f"\nServer error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\nError occurred: {e}")
    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")
        break