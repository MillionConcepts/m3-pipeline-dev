
## STEP 6: Dark Pedestal Shift Correction
# I am thinking this should be done using the dark field
# where each row of the dark is scale proportional to the 'dark pedestal effect'
# and added back to the image as a way of compensating for over-subtraction of dark
# when the values are suppressed at greater light. So we would use the avg value of
# the dark cols from the dark image proportional to the avg dark col values of the obs.
# I think that makes the most sense? The dark col signal is quite noisy though
# compared to the avg signal of the DSS image minus the dark cols.