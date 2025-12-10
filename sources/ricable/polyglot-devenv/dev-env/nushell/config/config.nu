$env.config = {
  show_banner: false
  table: {
    mode: rounded
    index_mode: always
    show_empty: true
  }
  completions: {
    case_sensitive: false
    quick: true
    partial: true
  }
  history: {
    max_size: 100_000
    sync_on_enter: true
  }
  error_style: "fancy"
  use_ansi_coloring: true
  edit_mode: emacs
}

# Custom aliases for development
alias ll = ls -la
alias dev = cd ~/dev
alias projects = ls ~/dev | where type == dir

# Git aliases
alias gs = git status
alias gp = git pull
alias gc = git commit -m

# Development environment shortcuts
alias py = cd python-env
alias ts = cd typescript-env
alias rs = cd rust-env
alias go = cd go-env
alias nu-env = cd nushell-env

# Devbox shortcuts
alias db = devbox
alias dbs = devbox shell
alias dbr = devbox run