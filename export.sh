#Export conda env to yml file
#Usage: ./export.sh <env_name> <yml_file_name>
#Example: ./export.sh myenv myenv.yml
mamba env export -n $1 > $2