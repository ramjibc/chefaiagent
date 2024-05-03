"""Class and function to mock tools used by Agent"""

from dataclasses import dataclass, Field
from datetime import datetime
from typing import List
from langchain_core.tools import tool


@dataclass
class Item:
    """food item class"""

    name: str
    quantity: int
    # best_by: datetime.date = datetime.today().date
    empty: bool


@dataclass
class SmartFridge:
    """Smart fridge class"""

    contents: List[Item]

    def get_quantity(self, item: Item) -> int:
        """return the current quantity of given item"""
        if item not in self.contents:
            return 0
        item_in_stock = [item_stock for item_stock in self.contents if item.name == item_stock.name][0]
        return item_in_stock.quantity

    def is_empty(self, item: Item) -> bool:
        """Ture if item is empty, false otherwise"""
        if item not in self.contents:
            return False
        item_in_stock = [item_stock for item_stock in self.contents if item.name == item_stock.name][0]
        return item_in_stock.empty


@tool
def check_item_in_fridge(item_name: str):
    """Check availability of given item in fridge"""
    # dumb init of fridge
    fridge = SmartFridge(contents=[Item(name="Honey", quantity=1, empty=False)])
    return fridge.is_empty(item=item_name)
    