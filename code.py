import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 0. ĐỌC DỮ LIỆU TỪ FILE CSV
# ==========================================
# Đọc toàn bộ file ZHVI.csv
df = pd.read_csv('ZHVI.csv', header=None)

# Cắt lấy đúng phần dữ liệu từ dòng 302 đến 314 trong Excel
df_filtered = df.iloc[301:314, :]

# Lấy dữ liệu: Cột K (Washington) là index 10, Cột S (Maine) là index 18
wa_prices = pd.to_numeric(df_filtered.iloc[:, 10]).values
me_prices = pd.to_numeric(df_filtered.iloc[:, 18]).values

# Tính toán số tháng tăng giá (cho Slide 2 và 4)
diffs_wa = np.diff(wa_prices)
diffs_me = np.diff(me_prices)

x_wa = np.sum(diffs_wa > 0) # Số tháng WA tăng giá
x_me = np.sum(diffs_me > 0) # Số tháng ME tăng giá
n_thang_so_sanh = len(diffs_wa) # Sẽ là 12 (vì có 13 tháng)

print("ĐANG XỬ LÝ DỮ LIỆU TỪ THÁNG 1/2025 ĐẾN 1/2026...\n")

# ==========================================
# SLIDE 1: KIỂM ĐỊNH 1 TRUNG BÌNH (Washington)
# ==========================================
# Giả thuyết: H0: mu = 580000 
mu0_s1 = 580000
n1_wa = len(wa_prices)
mean_wa = np.mean(wa_prices)
std_wa = np.std(wa_prices, ddof=1)

t_stat_1 = (mean_wa - mu0_s1) / (std_wa / math.sqrt(n1_wa))
p_value_1 = 2 * (1 - stats.t.cdf(abs(t_stat_1), df=n1_wa-1))

t_crit_1 = stats.t.ppf(1 - 0.05/2, df=n1_wa-1)
margin_1 = t_crit_1 * (std_wa / math.sqrt(n1_wa))
ci_lower_1, ci_upper_1 = mean_wa - margin_1, mean_wa + margin_1

conclusion_1 = "Reject H0" if p_value_1 < 0.05 else "Fail to reject H0"

print("=== 1. Conduct a hypothesis test and construct a confidence interval for the population mean. (Washington) ===")
print(f"KTC 95% cho mean: [{ci_lower_1:,.0f}, {ci_upper_1:,.0f}] | mean={mean_wa:,.0f}, s={std_wa:,.0f}, n={n1_wa}, t_alpha/2={t_crit_1:.4f}")
print(f"Kiểm định H0: mu = {mu0_s1}, t0 = {t_stat_1:.4f} \t\t\t Kết luận: {conclusion_1}\n")

# ==========================================
# SLIDE 2: KIỂM ĐỊNH 1 TỶ LỆ (Washington)
# ==========================================
# Giả thuyết: H0: p = 0.50 
p0_s2 = 0.50
p_hat_wa = x_wa / n_thang_so_sanh

z_crit_2 = stats.norm.ppf(1 - 0.05/2)
se_ci_2 = math.sqrt((p_hat_wa * (1 - p_hat_wa)) / n_thang_so_sanh)
ci_lower_2, ci_upper_2 = p_hat_wa - z_crit_2*se_ci_2, p_hat_wa + z_crit_2*se_ci_2

se_test_2 = math.sqrt((p0_s2 * (1 - p0_s2)) / n_thang_so_sanh)
z_stat_2 = (p_hat_wa - p0_s2) / se_test_2
p_value_2 = 2 * (1 - stats.norm.cdf(abs(z_stat_2)))

conclusion_2 = "Reject H0" if p_value_2 < 0.05 else "Fail to reject H0"

print("=== 2. Conduct a hypothesis test and construct a confidence interval for the population proportion. (Washington) ===")
print(f"Số tháng tăng/so sánh = {x_wa}/{n_thang_so_sanh} \t (p̂ = {p_hat_wa:.3f})")
print(f"KTC 95% cho p: [{ci_lower_2:.3f}, {ci_upper_2:.3f}]")
print(f"Kiểm định H0: p = {p0_s2:.2f} \t\t\t\t\t Kết luận: {conclusion_2}\n")

# ==========================================
# SLIDE 3: KIỂM ĐỊNH KHÁC BIỆT 2 TRUNG BÌNH (WA vs ME)
# ==========================================
mean_me = np.mean(me_prices)
std_me = np.std(me_prices, ddof=1)
n1_me = len(me_prices)

t_stat_3, p_value_3 = stats.ttest_ind(wa_prices, me_prices, equal_var=False)

# KTC Welch
var_wa, var_me = std_wa**2, std_me**2
df_3 = (var_wa/n1_wa + var_me/n1_me)**2 / ((var_wa/n1_wa)**2/(n1_wa-1) + (var_me/n1_me)**2/(n1_me-1))
t_crit_3 = stats.t.ppf(1 - 0.05/2, df_3)
margin_3 = t_crit_3 * math.sqrt(var_wa/n1_wa + var_me/n1_me)
diff_mean = mean_wa - mean_me

conclusion_3 = "Reject H0" if p_value_3 < 0.05 else "Fail to reject H0"

print("=== 3. Conduct a hypothesis test and construct a confidence interval for the difference between two population means. (WA vs ME) ===")
print(f"diff(mean) = {diff_mean:,.0f} | KTC 95%: [{diff_mean - margin_3:,.0f}, {diff_mean + margin_3:,.0f}]")
print(f"Kiểm định H0: mu_WA - mu_ME = 0, t0 = {t_stat_3:.3f} \t\t Kết luận: {conclusion_3}\n")

# ==========================================
# SLIDE 4: KIỂM ĐỊNH KHÁC BIỆT 2 TỶ LỆ (WA vs ME)
# ==========================================
p_hat_me = x_me / n_thang_so_sanh
diff_p = p_hat_wa - p_hat_me

se_ci_4 = math.sqrt((p_hat_wa*(1-p_hat_wa)/n_thang_so_sanh) + (p_hat_me*(1-p_hat_me)/n_thang_so_sanh))
z_crit_4 = stats.norm.ppf(1 - 0.05/2)
ci_lower_4, ci_upper_4 = diff_p - z_crit_4*se_ci_4, diff_p + z_crit_4*se_ci_4

p_pool = (x_wa + x_me) / (n_thang_so_sanh + n_thang_so_sanh)
se_test_4 = math.sqrt(p_pool * (1 - p_pool) * (2/n_thang_so_sanh))

if se_test_4 == 0:
    z_stat_4, p_value_4 = 0, 1
else:
    z_stat_4 = diff_p / se_test_4
    p_value_4 = 2 * (1 - stats.norm.cdf(abs(z_stat_4)))

conclusion_4 = "Reject H0" if p_value_4 < 0.05 else "Fail to reject H0"

print("=== 4. Conduct a hypothesis test and construct a confidence interval for the difference between two population proportions. ===")
print(f"p̂1 (WA) = {p_hat_wa:.3f}, p̂2 (ME) = {p_hat_me:.3f}, diff = {diff_p:.3f}")
print(f"z={z_stat_4:.3f}, p-value={p_value_4:.4f} | KTC 95% cho (p1 - p2): [{ci_lower_4:.3f}, {ci_upper_4:.3f}] \t Kết luận: {conclusion_4}\n")

# ==========================================
# SLIDE 5: PHÂN TÍCH HỒI QUY VÀ VẼ BIỂU ĐỒ
# ==========================================
thang_truc_x = np.arange(1, len(wa_prices) + 1)

res_wa = stats.linregress(thang_truc_x, wa_prices)
res_me = stats.linregress(thang_truc_x, me_prices)

print("=== 5. Regression analysis ===")
print(f"[Washington] y = {res_wa.intercept:.2f} + {res_wa.slope:.2f} * x;  R^2 = {res_wa.rvalue**2:.3f}, p-value = {res_wa.pvalue:.4f}, n = {len(thang_truc_x)}")
print(f"[Maine]      y = {res_me.intercept:.2f} + {res_me.slope:.2f} * x;  R^2 = {res_me.rvalue**2:.3f}, p-value = {res_me.pvalue:.4f}, n = {len(thang_truc_x)}")

# Vẽ hình
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Định nghĩa sẵn nhãn và vị trí để dùng chung cho cả 2 biểu đồ
nhan_thang_nam = ['1/2025', '2/2025', '3/2025', '4/2025', '5/2025', '6/2025', 
                  '7/2025', '8/2025', '9/2025', '10/2025', '11/2025', '12/2025', '1/2026']
vi_tri_cu = range(1, 14) 

# --- BIỂU ĐỒ 1: WASHINGTON (ax1) ---
ax1.plot(thang_truc_x, wa_prices, 'o-', label="Giá trị thực")
ax1.plot(thang_truc_x, res_wa.intercept + res_wa.slope*thang_truc_x, '-', label="Đường xu hướng")
# Chú ý: Đổi luôn p thành p-value trên tiêu đề biểu đồ cho đồng bộ
ax1.set_title(f"Xu hướng tuyến tính - Washington (R²={res_wa.rvalue**2:.3f}, p-value={res_wa.pvalue:.3f})")
ax1.set_xlabel("Thời gian")
ax1.set_ylabel("Giá trị")
ax1.legend()
ax1.set_xticks(vi_tri_cu)
ax1.set_xticklabels(nhan_thang_nam, rotation=45)

# --- BIỂU ĐỒ 2: MAINE (ax2) ---
ax2.plot(thang_truc_x, me_prices, 'o-', label="Giá trị thực")
ax2.plot(thang_truc_x, res_me.intercept + res_me.slope*thang_truc_x, '-', label="Đường xu hướng")
# Chú ý: Đổi luôn p thành p-value trên tiêu đề biểu đồ cho đồng bộ
ax2.set_title(f"Xu hướng tuyến tính - Maine (R²={res_me.rvalue**2:.3f}, p-value={res_me.pvalue:.3f})")
ax2.set_xlabel("Thời gian")
ax2.set_ylabel("Giá trị")
ax2.legend()
ax2.set_xticks(vi_tri_cu)
ax2.set_xticklabels(nhan_thang_nam, rotation=45)

# Đảm bảo layout không bị lẹm viền khi nghiêng chữ 45 độ
plt.tight_layout()

# Lưu thành ảnh
plt.savefig('bieu_do_mas291_wa_me.png', bbox_inches='tight')