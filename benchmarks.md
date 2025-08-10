# Baseline shakespeare, 100 steps. No tweaks

------------------------------------------------ benchmark: 1 tests -----------------------------------------------
Name (time in s) Min Max Mean StdDev Median IQR Outliers OPS Rounds Iterations

---

## test_train_shakespeare 40.0242 40.1074 40.0726 0.0308 40.0762 0.0345 2;0 0.0250 5 1

------------------------------------------------------ benchmark: 1 tests -----------------------------------------------------
Name (time in s) Min Max Mean StdDev Median IQR Outliers OPS Rounds Iterations

---

## test_train_shakespeare_update_only 39.9578 39.9833 39.9664 0.0100 39.9626 0.0109 1;0 0.0250 5 1

---------------------------------------------------------------------------------- benchmark: 2 tests ----------------------------------------------------------------------------------
Name (time in s) Min Max Mean StdDev Median IQR Outliers OPS Rounds Iterations

---

test_dataloader_iteration 1.2637 (1.0) 1.2713 (1.0) 1.2663 (1.0) 0.0031 (1.0) 1.2657 (1.0) 0.0043 (1.0) 1;0 0.7897 (1.0) 5 1
test_dataloader 1.2679 (1.00) 1.2832 (1.01) 1.2737 (1.01) 0.0057 (1.81) 1.2725 (1.01) 0.0043 (1.00) 2;1 0.7851 (0.99) 5 1

---

# Mmaped shakespeare, 100 steps. Added mmapped dataset
