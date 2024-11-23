from cashier.prompts.base_prompt import BasePrompt
from typing import Optional

class StateGuidelinePrompt(BasePrompt):

    def dynamic_prompt(  # type: ignore
        self,
        last_msg: Optional[str],
    ) -> str:
        return (
            "<state_guidelines>\n"
            "- Among the tools provided, there are functions for getting and updating the state defined in <state>. "
            "For state updates, you will have field specific update functions, whose names are `update_state_<field>` and where <field> is a state field.\n"
            "- You must update the state whenever applicable and as soon as possible. You cannot proceed to the next stage of the conversation without updating the state\n"
            "- Only you can update the state, so there is no need to udpate the state to the same value that had already been updated to in the past.\n"
            + (
                "- state updates can only happen in response to new messages (i.e. messages after <cutoff_msg>).\n"
                if last_msg
                else ""
            )
            + "</state_guidelines>\n"
        )
