from .constants import COLORS, DECORATORS, CYBER_BANNER
# Cyberpunk color scheme
COLORS = {
    "neon_blue": "\033[38;5;51m",
    "neon_pink": "\033[38;5;198m", 
    "neon_green": "\033[38;5;46m",
    "neon_yellow": "\033[38;5;226m",
    "neon_red": "\033[38;5;196m",
    "neon_purple": "\033[38;5;165m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blink": "\033[5m"
}

# Cyberpunk decorators
DECORATORS = {
    "box_top": f"{COLORS['neon_blue']}╔{'═'*60}╗{COLORS['reset']}",
    "box_bottom": f"{COLORS['neon_blue']}╚{'═'*60}╝{COLORS['reset']}",
    "box_line": f"{COLORS['neon_blue']}║{COLORS['reset']}",
    "arrow": f"{COLORS['neon_pink']}[►]{COLORS['reset']}"
}

CYBER_BANNER = f"""
{COLORS['neon_blue']}
    █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     
   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝
{COLORS['neon_pink']}   ███████╗███████╗ ██████╗██╗   ██╗██████╗ ██╗████████╗██╗   ██╗
   ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝
   ███████╗█████╗  ██║     ██║   ██║██████╔╝██║   ██║    ╚████╔╝ 
   ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██║   ██║     ╚██╔╝  
   ███████║███████╗╚██████╗╚██████╔╝██║  ██║██║   ██║      ██║   
   ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝   
{COLORS['reset']}
{COLORS['neon_purple']}[ {COLORS['blink']}AI-Powered Security Scanner & Auto-Fix Pipeline{COLORS['reset']}{COLORS['neon_purple']} ]{COLORS['reset']}
{COLORS['neon_green']}[ Created by rUv, cause he could. ]{COLORS['reset']}

"""
