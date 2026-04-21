from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

URBAN_DISTRICTS = [
    "Ba Đình", "Hoàn Kiếm", "Tây Hồ", "Long Biên", "Cầu Giấy",
    "Đống Đa", "Hai Bà Trưng", "Hoàng Mai", "Thanh Xuân", 
    "Nam Từ Liêm", "Bắc Từ Liêm", "Hà Đông"
]

RURAL_DISTRICTS = [
    "Sơn Tây", "Ba Vì", "Chương Mỹ", "Đan Phượng", "Đông Anh",
    "Gia Lâm", "Hoài Đức", "Mê Linh", "Mỹ Đức", "Phú Xuyên", 
    "Phúc Thọ", "Quốc Oai", "Sóc Sơn", "Thạch Thất", "Thanh Oai", 
    "Thanh Trì", "Thường Tín", "Ứng Hòa"
]

ALL_DISTRICTS = URBAN_DISTRICTS + RURAL_DISTRICTS

def extract_district(text: str) -> str | None:
    """Extracts a Hanoi district from project name text."""
    if not isinstance(text, str):
        return None
    
    # Try exact match first (case-insensitive)
    for district in ALL_DISTRICTS:
        if district.lower() in text.lower():
            return district
            
    # Try removing prefixes like 'Quận ', 'Huyện '
    prefixes = ["Quận ", "Huyện ", "thị xã "]
    for prefix in prefixes:
        for district in ALL_DISTRICTS:
            if (prefix.lower() + district.lower()) in text.lower():
                return district
                
    return "Hà Nội (Chung)"

def process_projects(csv_path: Path) -> pd.DataFrame:
    """Processes the projects CSV to count density per district."""
    if not csv_path.exists():
        return pd.DataFrame(columns=["district", "project_count", "projects"])
    
    df = pd.read_csv(csv_path)
    df["district"] = df["project_name"].apply(extract_district)
    
    # Group by district
    density = df.groupby("district").agg({
        "project_name": "count",
        "status": lambda x: list(x)
    }).reset_index()
    
    density.columns = ["district", "project_count", "statuses"]
    
    # Also keep a list of project names for detail display
    details = df.groupby("district")["project_name"].apply(list).to_dict()
    density["project_list"] = density["district"].map(details)
    
    return density

def get_district_risk(district: str, density_df: pd.DataFrame) -> dict:
    """Returns risk info for a specific district."""
    row = density_df[density_df["district"] == district]
    if row.empty:
        return {"level": "Low", "count": 0, "projects": []}
    
    count = row.iloc[0]["project_count"]
    projects = row.iloc[0]["project_list"]
    
    if count >= 3:
        level = "High"
    elif count >= 1:
        level = "Moderate"
    else:
        level = "Low"
        
    return {"level": level, "count": count, "projects": projects}
