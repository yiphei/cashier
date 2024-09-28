from openai import OpenAI
from db_functions import get_menu_items_options, get_menu_item_from_name
import json
from dataclasses import asdict
client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

SYSTEM_PROMPT = "You are a cashier working for the coffee shop Heaven Coffee. Customers come to you to place orders. " \
                "Your job is to take their orders, answer reasonable questions about the shop & menu only, and assist " \
                "them with any issues they may have about their orders. You are not responsible for anything else, " \
                "so you must refuse to engage in anything unrelated"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_menu_items_options",
            "description": "Get all the options available for the menu item. " \
                            "Most customers either don't provide a complete order (i.e. not specifying required options like size)" \
                            "or are not aware of all the options available for a menu item. It is your job to help them with both cases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_item_id": {
                        "type": "string",
                        "description": "The menu item id used in the db."
                    }
                },
                "required": ["menu_item_id"],
                "additionalProperties": False
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_menu_item_from_name",
            "description": "Get the menu item given the name string of the menu item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_item_name": {
                        "type": "string",
                        "description": "The menu item name"
                    }
                },
                "required": ["menu_item_name"],
                "additionalProperties": False
            }
        }
    }
]

if __name__ == "__main__":

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": "hi, welcome to Heaven Coffee"}],
    print("Assistant: hi, welcome to Heaven Coffee")
    need_user_input = True

    while True:
        if need_user_input:
            # Read user input from stdin
            user_input = input("You: ")
            # If user types 'exit', break the loop and end the program
            if user_input.lower() == 'exit':
                print("Exiting chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})

        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
    
        response = chat_completion.choices[0]
        finish_reason = response.finish_reason
        if finish_reason == "stop":
            response_msg = response.message.content
            print("Assistant: ", response_msg)
            messages.append({"role": "assistant", "content": response_msg})
            
            need_user_input = True
        elif finish_reason == "tool_calls":
            tool_call_message = response.message
            function_name = response.message.tool_calls[0].function.name
            fuction_args = json.loads(response.message.tool_calls[0].function.arguments)
            tool_call_id = response.message.tool_calls[0].id

            if function_name == "get_menu_item_from_name":
                menu_item = get_menu_item_from_name(**fuction_args)
                content = asdict(menu_item)
            else:
                size_to_default_options_map, size_to_available_options = get_menu_items_options(**fuction_args)
                content = {
                    "size_to_default_options_map": size_to_default_options_map,
                    "size_to_available_options": size_to_available_options
                }

            function_call_result_msg = {
                "role": "tool",
                "content": json.dumps(content),
                "tool_call_id": tool_call_id
            }
            messages.append(tool_call_message)
            messages.append(function_call_result_msg)

            need_user_input = False
