# Simulating power analysis for the current study
# Making conclusions on participant level

# Null hypothesis (control): p = 0.6
# Alt hypothesis experimental_1 (only audio) p = 0.7
# Alt hypothesis experimental_2 (audio + lyrics) p = 0.8

# Specify:
# Number of trials
N = 600
# Number of replications
reps = 10000

# First test (H1): Control vs Experimental (audio-only)
count <- numeric(reps)
for(i in 1:reps){
  control <- rbinom(round(N/3),1, 0.6)
  experimental_1 <- rbinom(round(N/3), 1, 0.7)
  count[i] <- prop.test(x = c(sum(control), sum(experimental_1)), n = c(N/3,N/3), alternative="less")$p.value
}
print("Power Analysis results for H1: Control vs Experimental (audio-only)")
prop.table(table(count<.05))

# Second test (H2): Control vs Experimental (multimodal)
for(i in 1:reps){
  control <- rbinom(round(N/3),1, 0.6)
  experimental_2 <- rbinom(round(N/3), 1, 0.8)
  count[i] <- prop.test(x = c(sum(control), sum(experimental_2)), n = c(N/3,N/3), alternative="less")$p.value
}
print("Power Analysis results for H1: Control vs Experimental (audio-only)")
prop.table(table(count<.05))
# Third t-test: Experimental (only audio) vs Experimental (audio + lyrics)
for(i in 1:reps){
  experimental_1 <- rbinom(round(N/3),1, 0.7)
  experimental_2 <- rbinom(round(N/3), 1, 0.8)
  count[i] <- prop.test(x = c(sum(experimental_1), sum(experimental_2)), n = c(N/3,N/3), alternative="less")$p.value
}
print("Power Analysis results for H3: Experimental (audio-only) vs Experimental (multimodal)")
prop.table(table(count<.05))

