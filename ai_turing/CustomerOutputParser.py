from pydantic import BaseModel
from typing import Type
from langchain_core.output_parsers import BaseOutputParser


class CustomerOutputParser(BaseOutputParser[BaseModel]):
    
    pydantic_object: Type[T]


    def parse(self, text: str) -> BaseModel:
        return self.pydantic_object.model_validate_json(text)

    def get_format_instructions(self) -> str:
        return self.pydantic_object.model_json_schema()

    @property
    def _type(self) -> str:
        return "customer_output_parser"