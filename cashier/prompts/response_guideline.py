from cashier.prompts.base_prompt import BasePrompt


class ResponseGuidelinePrompt(BasePrompt):

    def dynamic_prompt(
        self,
        is_voice_only,
    ):
        return (
            "<response_guidelines>\n"
            + (
                "- because your responses will be converted to speech, "
                "you must respond in a conversational way: natural, easy to understand when converted to speech, and generally concise and brief (no long responses).\n"
                "- AVOID using any rich text formatting like hashtags, bold, italic, bullet points, numbered points, headers, etc.\n"
                "- When responding to customers, AVOID providing unrequested information.\n"
                if is_voice_only
                else ""
            )
            + "- If a response to a request is naturally long, then either ask claryfing questions to further refine the request, "
            "summarize the response, or break down the response in many separate responses.\n"
            "- Overall, try to be professional, polite, empathetic, and friendly\n"
            "</response_guidelines>\n"
        )
