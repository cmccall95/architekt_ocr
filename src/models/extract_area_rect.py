
from src.models.column_name import ColumnName

class ExtractAreaRect:
    def __init__(self, column_name: ColumnName, relative_x1: float, relative_x2: float, relative_y1: float, relative_y2: float):
        self.column_name = column_name
        self.relative_x1 = relative_x1
        self.relative_x2 = relative_x2
        self.relative_y1 = relative_y1
        self.relative_y2 = relative_y2
    
    @classmethod
    def fromJson(cls, json_obj):
        column_name = ColumnName(json_obj["columnName"])
        relative_x1 = json_obj["relativeX1"]
        relative_x2 = json_obj["relativeX2"]
        relative_y1 = json_obj["relativeY1"]
        relative_y2 = json_obj["relativeY2"]
        return cls(column_name, relative_x1, relative_x2, relative_y1, relative_y2)