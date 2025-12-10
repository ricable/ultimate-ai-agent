#! /usr/bin/env bash
#
# Create developement environment.
#

main() {
    init
    echo -e "${GREEN}Installing SPARC CLI dependencies...${NC}"

    # Change to the sparc directory
    cd "$SCRIPT_DIR"

    venv
    deps

    echo -e "\n${GREEN}Installation complete!${NC}"
    echo 'First, always load your environment: source .venv/bin/activate'
    echo 'Then you can use sparc. Try: sparc --help'
}

init() {
    set -o errexit
    set -o nounset
    set -o pipefail
    if [[ "${DEBUG:-}" ]]
    then
        PS4='\r$(PS4func $LINENO)'
        set -o xtrace
    fi

    # Colors
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    # Get script directory
    SCRIPT_DIR=$(here)
}

venv() {
    if [[ ! -e .venv/bin/activate ]]
    then
        python3.12 -m venv .venv
        source .venv/bin/activate
        pip list --format=freeze \
          |grep -oE '^[^=]+' \
          |xargs pip install --upgrade
    else
        source .venv/bin/activate
    fi
}

deps() {

    pip install --requirement requirements-dev.txt
    pip install --editable .

    if command -v rg &>/dev/null
    then return 0
    fi
    # Install ripgrep if not present
    if command -v apt-get &>/dev/null
    then
        sudo apt-get update
        sudo apt-get install --yes ripgrep

    elif command -v dnf &>/dev/null
    then
        sudo dnf install --yes ripgrep

    elif command -v yum &>/dev/null
    then
        sudo yum install --yes ripgrep

    elif command -v brew &>/dev/null
    then
        brew install ripgrep

    else
        echo "Please install ripgrep manually: https://github.com/BurntSushi/ripgrep#installation"
        # TODO: rustup then cargo binstall ripgrep
    fi
}

here() {
    local dirn
    dirn=$(dirname -- "${BASH_SOURCE[0]}")
    cd -- "$dirn"
    pwd
}

PS4func() {
    local lineno="$1"
    local i f=''
    local c="\033[0;36m" y="\033[0;33m" n="\033[0m"
    local d=$((${#FUNCNAME[@]}-2))

    if [[ $lineno == 1 ]]
    then lineno=0
    fi

    for ((i=d; i>0; i--))
    do printf -v f "%s%s()" "$f" "${FUNCNAME[i]}"
    done

    printf "$y%s:%04d$c%s$n " "${BASH_SOURCE[1]##*/}" "$lineno" "$f"
}

main
exit 0
