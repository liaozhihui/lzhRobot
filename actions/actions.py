from typing import Dict, Text, Any, List, Union
import re
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction


class ValidateChargeForm(FormValidationAction):
    """Example of a form validation action."""

    def name(self) -> Text:
        return "validate_charge_form"



    def validate_money(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate cuisine value."""
        print("money_value:",value)
        if value.endswith("元") or not value.endswith("块"):
            result = re.findall("(\d*)[元|块]", value)[0]

            return {"money": result}
        else:
            dispatcher.utter_message(response="utter_wrong_money")
            # validation failed, set slot to None
            return {"money": None}




    def validate_phone_number(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate num_people value."""
        print("validate_phone_number:",value)

        if len(value)!=11:
            dispatcher.utter_message(response="utter_wrong_phone")

            return {"phone_number":None}
        return {"phone_number": value}


