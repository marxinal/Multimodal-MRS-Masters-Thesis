# Simulating power analysis for the future study
# Making conclusions on the population level

# Null hypothesis (control): p = 0.6
# Alt hypothesis experimental_1 (only audio) p = 0.7
# Alt hypothesis experimental_2 (audio + lyrics) p = 0.8
# Specifying number of trials per participant and the number of participants

# Specify:
# Number of trials
N = 160
# Number of replications
reps = 10000
# Number of participants
n_participants = 30

# First test (H1): Control vs Experimental (audio-only)
count <- numeric(reps)
control <- c()
experimental <- c()

for(i in 1:reps){
  for(j in 1:n_participants){
    control[j] <- sum(rbinom(round(N/3),1, 0.6))
    experimental[j] <- sum(rbinom(round(N/3), 1, 0.7))
  }
  control_all <- sum(control)
  experimental_all <- sum(experimental)
  count[i] <- prop.test(x = c(sum(control_all),
                              sum(experimental_all)),
                        n = c((N/3)*n_participants,(N/3)*n_participants),
                        alternative="less")$p.value
}
print("Power Analysis results for H1: Control vs Experimental (audio-only)")
prop.table(table(count<.05))

# Second test (H2): Control vs Experimental (multimodal)
control <- c()
experimental_1 <- c()

for(i in 1:reps){
  for(j in 1:n_participants){
    control[j] <- sum(rbinom(round(N/3),1, 0.6))
    experimental_1[j] <- sum(rbinom(round(N/3), 1, 0.8))
  }
  control_all <- sum(control)
  experimental_all_1 <- sum(experimental_1)
  count[i] <- prop.test(x = c(sum(control_all),
                              sum(experimental_all_1)),
                        n = c((N/3)*n_participants,(N/3)*n_participants),
                        alternative="less")$p.value
}
print("Power Analysis results for H1: Control vs Experimental (audio-only)")
prop.table(table(count<.05))

# Third test (H3): Experimental (audio-only) vs Experimental (multimodal)
experimental_1 <- c()
experimental_2 <- c()
for(i in 1:reps){
  for(j in 1:n_participants){
    experimental_1[j] <- sum(rbinom(round(N/3),1, 0.7))
    experimental_2[j] <- sum(rbinom(round(N/3), 1, 0.8))
  }
  experimental_all_1 <- sum(experimental_1)
  experimental_all_2 <- sum(experimental_2)
  count[i] <- prop.test(x = c(sum(experimental_all_1),
                              sum(experimental_all_2)),
                        n = c((N/3)*n_participants,(N/3)*n_participants),
                        alternative="less")$p.value
}
print("Power Analysis results for H3: Experimental (audio-only) vs Experimental (multimodal)")
prop.table(table(count<.05))