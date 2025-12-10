#!/usr/bin/env nu

# Sync devbox.json files with common.json

def main [] {
    let common_config = (open ../common.json)
    let common_packages = $common_config.packages.devbox

    let devbox_files = (glob "*-env/devbox.json")

    for file in $devbox_files {
        let devbox_config = (open $file)
        let updated_packages = ($devbox_config.packages | append $common_packages | uniq)
        let updated_config = ($devbox_config | update packages $updated_packages)
        $updated_config | to json --indent 4 | save $file --force
        echo $"Synced ($file)"
    }
}