"""Class and function to mock tools used by Agent"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from langchain_core.tools import tool


@dataclass
class Item:
    """food item class"""

    name: str
    quantity: int = field(compare=False, default=1)
    best_by: datetime.date = field(default=datetime.today().date, compare=False)
    empty: bool = field(default=False, compare=False)


@dataclass
class SmartFridge:
    """Smart fridge class"""

    contents: List[Item]

    def get_quantity(self, item: Item) -> int:
        """return the current quantity of given item"""
        if item not in self.contents:
            return 0
        return item.quantity

    def is_empty(self, item: Item) -> bool:
        """Ture if item is empty, false otherwise"""
        if item not in self.contents:
            return False
        return item.empty


@tool
def check_item_in_fridge(item_name: str):
    """Check availability of given item in fridge"""
    # dumb init of fridge
    fridge = SmartFridge(contents=[Item(name="Honey"), Item(name="butter")])
    return fridge.is_empty(item=Item(name=item_name))
