#!/bin/bash
[[ $_ == $0 ]]; is_sourced=$?
my_dir=$(cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd)

repo_id=$( head -n1 "$my_dir/../REPO_ID" )
repo_id=${repo_id/ /-}


if [ -z "$repo_id" ] || [ "$repo_id" = "pparker" ]
then
	echo "[!] Please specify the repository id in the file REPO_ID"
	if (( $is_sourced == 1 ));
	then
		return 1
	fi
else
	ln_env="$my_dir/../env"
	env="$( echo ~ )/.local/env/student-$repo_id"
	
	if [ -e "$env" ] && [ -e "$env/bin/activate" ]
	then
		source "$env/bin/activate"
	else
		python3 -m venv "$env"
		if [ $? -ne 0 ]
		then
			echo "[*] Install python/ virtual environments"
			echo "    apt install python3 python3-pip python3-venv"
			[[ "${BASH_SOURCE[0]}" != "${0}" ]] && return || exit
		fi
		source "$env/bin/activate"
		pip install --upgrade pip
		pip install -r "$my_dir/../requirements.txt"
	
		grep -P '\#' "$my_dir/../requirements.txt"
	fi
	
	rm -R "$ln_env" > /dev/null 2>&1
	ln -s "$env" "$ln_env"
fi