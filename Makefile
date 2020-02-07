dirs := ./lib/

default:
	@$(foreach var, $(dirs), make -C $(var);)

clean:
	@$(foreach var, $(dirs), make -C $(var) clean;)