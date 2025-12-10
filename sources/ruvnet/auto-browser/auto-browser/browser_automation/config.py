from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

class Selector(BaseModel):
    """Configuration for a specific element selector"""
    css: str
    attribute: Optional[str] = None
    multiple: bool = False
    required: bool = True
    description: Optional[str] = None

class SiteConfig(BaseModel):
    """Configuration for a specific site template"""
    name: str
    description: Optional[str] = None
    url_pattern: str
    selectors: Dict[str, Union[str, Selector]]
    wait_for: Optional[str] = None
    output_format: str = "markdown"
    delay: float = 2.0
    
    @validator('selectors')
    def validate_selectors(cls, v):
        """Convert string selectors to Selector objects"""
        processed = {}
        for key, value in v.items():
            if isinstance(value, str):
                processed[key] = Selector(css=value)
            elif isinstance(value, dict):
                processed[key] = Selector(**value)
            else:
                processed[key] = value
        return processed

class Config(BaseModel):
    """Main configuration"""
    sites: Dict[str, SiteConfig]
    output_dir: Path = Field(default=Path("output"))
    default_site: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('sites')
    def validate_default_site(cls, v, values):
        """Ensure default_site is valid if specified"""
        if 'default_site' in values and values['default_site']:
            if values['default_site'] not in v:
                raise ValueError(f"Default site '{values['default_site']}' not found in sites")
        return v

def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file"""
    import yaml
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
            
    try:
        return Config(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration format: {e}")

def create_example_config(path: Path):
    """Create an example configuration file"""
    example_config = {
        "sites": {
            "clinical_trials": {
                "name": "EU Clinical Trials",
                "description": "Extract data from EU Clinical Trials website",
                "url_pattern": "https://euclinicaltrials.eu/ctis-public/view/{trial_id}",
                "selectors": {
                    "title": {
                        "css": "h1.trial-title",
                        "required": True,
                        "description": "Trial title"
                    },
                    "summary": {
                        "css": "div.trial-summary",
                        "multiple": False,
                        "description": "Trial summary"
                    },
                    "details": {
                        "css": "div.trial-details div",
                        "multiple": True,
                        "description": "Trial detail sections"
                    }
                },
                "wait_for": "div.trial-content",
                "output_format": "markdown",
                "delay": 2.0
            },
            "wiki": {
                "name": "Wikipedia",
                "description": "Extract content from Wikipedia articles",
                "url_pattern": "https://en.wikipedia.org/wiki/{title}",
                "selectors": {
                    "title": "h1#firstHeading",
                    "content": {
                        "css": "div#mw-content-text p",
                        "multiple": True,
                        "description": "Article paragraphs"
                    }
                },
                "output_format": "markdown",
                "delay": 1.0
            }
        },
        "output_dir": "output",
        "default_site": "clinical_trials"
    }
    
    import yaml
    with open(path, 'w') as f:
        yaml.dump(example_config, f, sort_keys=False, indent=2)
