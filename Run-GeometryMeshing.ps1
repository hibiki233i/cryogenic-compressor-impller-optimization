# ==============================================================================
# 离心压缩机自动化几何与网格生成模块 
# ==============================================================================

param (
    [string]$WorkingDir = "F:\optimazition\Runs\Run_001",
   [double]$d1s, [double]$dH, [double]$beta1hb, [double]$beta1sb, [double]$d2, 
    [double]$b2, [double]$beta2hb, [double]$beta2sb, [double]$Lz, [double]$t, 
    [double]$TipClear, [double]$nBl, [double]$rake_te_s,
    [double]$mFlow, [double]$N_rpm, [double]$alpha0
)

$ErrorActionPreference = "Stop"

function Get-LogWarningInfo {
    param(
        [string]$LogPath
    )

    $info = [ordered]@{
        warn_sweep = $false
        warn_overlap = $false
        warn_internal_blade_thickness = $false
        overlap_limit_deg = $null
        warning_messages = @()
    }

    if (-not (Test-Path $LogPath)) {
        return $info
    }

    $logContent = Get-Content -Path $LogPath -Raw
    $warningLines = Select-String -InputObject $logContent -Pattern "\[WARN\].*" -AllMatches |
        ForEach-Object { $_.Matches } |
        ForEach-Object { $_.Value.Trim() }

    foreach ($line in $warningLines) {
        if ($line -like "*Very high tangential leading edge sweep angle*") {
            $info.warn_sweep = $true
        }
        if ($line -like "*Overlapping of adjacent blades might be too low*") {
            $info.warn_overlap = $true
            $m = [regex]::Match($line, "Δφ=([0-9.]+)°")
            if ($m.Success) {
                $info.overlap_limit_deg = [double]$m.Groups[1].Value
            }
        }
        if ($line -like "*Internal blade thickness is lower than specified*") {
            $info.warn_internal_blade_thickness = $true
        }
    }

    $info.warning_messages = $warningLines
    return $info
}

function Get-GeometrySummary {
    param(
        [string]$CftResPath,
        [string]$LogPath
    )

    $warningInfo = Get-LogWarningInfo -LogPath $LogPath
    $summary = [ordered]@{
        source_cft_res = $CftResPath
        source_log = $LogPath
        parsed = $false
        warn_sweep = $warningInfo.warn_sweep
        warn_overlap = $warningInfo.warn_overlap
        warn_internal_blade_thickness = $warningInfo.warn_internal_blade_thickness
        overlap_limit_deg_from_log = $warningInfo.overlap_limit_deg
        warning_messages = $warningInfo.warning_messages
    }

    if (-not (Test-Path $CftResPath)) {
        return $summary
    }

    $xml = [xml](Get-Content -Path $CftResPath -Raw)

    $nBlNode = $xml.SelectSingleNode("//Updates//nBl")
    $phiLeNodes = $xml.SelectNodes("//Output//phiLE/Value")
    $phiTeNodes = $xml.SelectNodes("//Output//phiTE/Value")
    $overlapNodes = $xml.SelectNodes("//Output//TMeanLine/OverlapAngle")
    $beta1Nodes = $xml.SelectNodes("//Updates//Beta1/Value")
    $beta2Nodes = $xml.SelectNodes("//Updates//Beta2/Value")
    $leanLeNodes = $xml.SelectNodes("//Output//LeanLE/Value")
    $rakeTeNodes = $xml.SelectNodes("//Output//RakeTE/Value")

    if ($null -eq $nBlNode -or $phiLeNodes.Count -lt 2 -or $phiTeNodes.Count -lt 2 -or $overlapNodes.Count -lt 2) {
        return $summary
    }

    $nBlVal = [int][math]::Round([double]$nBlNode.InnerText)
    $pitchDeg = 360.0 / $nBlVal

    $phiLeHubDeg = [double]$phiLeNodes[0].InnerText * 180.0 / [math]::PI
    $phiLeShroudDeg = [double]$phiLeNodes[1].InnerText * 180.0 / [math]::PI
    $phiTeHubDeg = [double]$phiTeNodes[0].InnerText * 180.0 / [math]::PI
    $phiTeShroudDeg = [double]$phiTeNodes[1].InnerText * 180.0 / [math]::PI

    $dphiHubDeg = $phiTeHubDeg - $phiLeHubDeg
    $dphiShroudDeg = $phiTeShroudDeg - $phiLeShroudDeg

    $overlapHubDeg = [double]$overlapNodes[0].InnerText * 180.0 / [math]::PI
    $overlapShroudDeg = [double]$overlapNodes[1].InnerText * 180.0 / [math]::PI

    $overlapFactorHub = $dphiHubDeg / $pitchDeg
    $overlapFactorShroud = $dphiShroudDeg / $pitchDeg
    $overlapFactorMin = [math]::Min($overlapFactorHub, $overlapFactorShroud)

    $summary.parsed = $true
    $summary.nBl = $nBlVal
    $summary.pitch_deg = $pitchDeg
    $summary.phiLE_hub_deg = $phiLeHubDeg
    $summary.phiLE_shroud_deg = $phiLeShroudDeg
    $summary.phiTE_hub_deg = $phiTeHubDeg
    $summary.phiTE_shroud_deg = $phiTeShroudDeg
    $summary.dphi_hub_deg = $dphiHubDeg
    $summary.dphi_shroud_deg = $dphiShroudDeg
    $summary.overlap_angle_hub_deg = $overlapHubDeg
    $summary.overlap_angle_shroud_deg = $overlapShroudDeg
    $summary.overlap_factor_hub = $overlapFactorHub
    $summary.overlap_factor_shroud = $overlapFactorShroud
    $summary.overlap_factor_min = $overlapFactorMin

    if ($beta1Nodes.Count -ge 2) {
        $beta1HubDeg = [double]$beta1Nodes[0].InnerText * 180.0 / [math]::PI
        $beta1ShroudDeg = [double]$beta1Nodes[1].InnerText * 180.0 / [math]::PI
        $summary.beta1_hub_deg = $beta1HubDeg
        $summary.beta1_shroud_deg = $beta1ShroudDeg
        $summary.beta1_diff_deg = $beta1HubDeg - $beta1ShroudDeg
    }

    if ($beta2Nodes.Count -ge 2) {
        $beta2HubDeg = [double]$beta2Nodes[0].InnerText * 180.0 / [math]::PI
        $beta2ShroudDeg = [double]$beta2Nodes[1].InnerText * 180.0 / [math]::PI
        $summary.beta2_hub_deg = $beta2HubDeg
        $summary.beta2_shroud_deg = $beta2ShroudDeg
        $summary.beta2_diff_deg = $beta2HubDeg - $beta2ShroudDeg
    }

    if ($leanLeNodes.Count -ge 2) {
        $leanLeHubDeg = [double]$leanLeNodes[0].InnerText * 180.0 / [math]::PI
        $leanLeShroudDeg = [double]$leanLeNodes[1].InnerText * 180.0 / [math]::PI
        $summary.leanLE_hub_deg = $leanLeHubDeg
        $summary.leanLE_shroud_deg = $leanLeShroudDeg
        $summary.leanLE_diff_deg = $leanLeHubDeg - $leanLeShroudDeg
    }

    if ($rakeTeNodes.Count -ge 2) {
        $summary.rakeTE_hub_deg = [double]$rakeTeNodes[0].InnerText * 180.0 / [math]::PI
        $summary.rakeTE_shroud_deg = [double]$rakeTeNodes[1].InnerText * 180.0 / [math]::PI
    }

    return $summary
}

# ==============================================================================
# 0. 环境与路径准备
# ==============================================================================                                                                                                                                                                                                                            
# 软件执行路径
$CFTurbo_Exe   = "C:\Program Files\CFturbo 2025.2.2\CFturbo.exe"
$TurboGrid_Exe = "D:\ANSYS Inc\v251\TurboGrid\bin\cfxtg.exe"

# 模板文件路径 (直接使用 CFturbo 和 TurboGrid 原生导出的文件)
$CFT_Template  = "F:\optimazition\Templates\BaseModel.cft-batch" 
$TGS_Template  = "F:\optimazition\Templates\BaseMeshing.tst"

# 当前计算步的工作文件
$Current_CFT   = Join-Path $WorkingDir "run_cfturbo.cft-batch"
$Current_TGS   = Join-Path $WorkingDir "run_turbogrid.tst"
$Export_Hub     = Join-Path $WorkingDir "Impeller_hub.curve"
$Export_Shroud  = Join-Path $WorkingDir "Impeller_shroud.curve"
$Export_Profile = Join-Path $WorkingDir "Impeller_profile.curve"
$Export_Mesh    = Join-Path $WorkingDir "Impeller_Mesh.gtm"
$Geometry_Summary = Join-Path $WorkingDir "geometry_summary.json"
# 在生成网格之前，把原始 cft 复制到当前计算目录
Copy-Item -Path "F:\optimazition\Templates\0908-2.cft" -Destination $WorkingDir -Force


Write-Host ">>> 开始执行几何与网格生成流水线..." -ForegroundColor Cyan

if (Test-Path $Export_Mesh) { Remove-Item $Export_Mesh -Force }

try {
Write-Host "--> [1/3] 解析并修改 CFturbo XML 参数..."
    
    # 将模板加载为标准 XML 对象，避免一切正则匹配导致的格式崩溃
    $xml = [xml](Get-Content -Path $CFT_Template -Raw)

    # 1. 基础几何尺寸
    $xml.SelectSingleNode("//dS").'#text' = $d1s.ToString("F5")
    $xml.SelectSingleNode("//d2").'#text' = $d2.ToString("F5")
    $xml.SelectSingleNode("//b2").'#text' = $b2.ToString("F5")
    $xml.SelectSingleNode("//DeltaZ").'#text' = $Lz.ToString("F5") # 对应轴向长度
    # 叶片数
    $xml.SelectSingleNode("//nBl").'#text' = $nBl.ToString()
    #轮毂直径
    $xml.SelectSingleNode("//dH").'#text' = $dH.ToString("F5")
    # 2. 间隙与厚度 (一对多赋值)
    $xml.SelectSingleNode("//xTipInlet").'#text' = $TipClear.ToString("F5")
    $xml.SelectSingleNode("//xTipOutlet").'#text' = $TipClear.ToString("F5")
    $nBl_Int = [math]::Round($nBl)
    $xml.SelectSingleNode("//sLEH").'#text' = $t.ToString("F5")
    $xml.SelectSingleNode("//sLES").'#text' = $t.ToString("F5")
    $xml.SelectSingleNode("//sTEH").'#text' = $t.ToString("F5")
    $xml.SelectSingleNode("//sTES").'#text' = $t.ToString("F5")

    # 转换为弧度
    $beta2hb_rad = $beta2hb * [math]::Pi / 180.0
    $beta2sb_rad = $beta2sb * [math]::Pi / 180.0

# 独立赋值
    $xml.SelectSingleNode("//Beta2/Value[@Index='0']").'#text' = $beta2hb_rad.ToString("F6") # Hub
    $xml.SelectSingleNode("//Beta2/Value[@Index='1']").'#text' = $beta2sb_rad.ToString("F6") # Shroud
    # 2. 加入出口后掠角 (仅修改 Shroud 处，Index='1')
    $rake_te_s_rad = $rake_te_s * [math]::Pi / 180.0
    # 注意：如果 XML 中没有暴露 Index='0'，也可以同时给 Hub 赋 0 值
    $xml.SelectSingleNode("//RakeTE/Value[@Index='1']").'#text' = $rake_te_s_rad.ToString("F6")

    # 3. 角度转换 (Python 传进来的是度，CFturbo 需要的是弧度)
    $beta1hb_rad = $beta1hb * [math]::Pi / 180.0
    $beta1sb_rad = $beta1sb * [math]::Pi / 180.0


    # 进口角 Beta1 (Index='0' 是 Hub，Index='1' 是 Shroud)
    $xml.SelectSingleNode("//Beta1/Value[@Index='0']").'#text' = $beta1hb_rad.ToString("F6")
    $xml.SelectSingleNode("//Beta1/Value[@Index='1']").'#text' = $beta1sb_rad.ToString("F6")
    
   

    # 4. 运行工况常数
    $xml.SelectSingleNode("//mFlow").'#text' = $mFlow.ToString("F6")
    # nRot 在 XML 里的单位是 Hz (/s)，所以要除以 60
    $nRot_Hz = $N_rpm / 60.0 
    $xml.SelectSingleNode("//nRot").'#text' = $nRot_Hz.ToString("F6")

    # 最终保存为新的工作文件
    $xml.Save($Current_CFT)

    # ==============================================================================
# [2/3] 运行 CFturbo 并在运行前准备好基础工程文件
# ==============================================================================
Write-Host "--> [2/3] 准备运行 CFturbo..."

# 1. 强制将你 Templates 目录下的母本 .cft 文件复制到当前的 Run_xxx 文件夹里
# (注意：请确认你的母本文件是不是叫这个名字，如果不是请修改)
$Source_CFT_Model = "F:\optimazition\Templates\0908-2.cft"
$Target_CFT_Model = Join-Path $WorkingDir "0908-2.cft"

if (Test-Path $Source_CFT_Model) {
    Copy-Item -Path $Source_CFT_Model -Destination $Target_CFT_Model -Force
} else {
    Write-Warning "!!! 找不到母本工程文件: $Source_CFT_Model"
    exit 2
}

# 2. 启动 CFturbo
$cftProcess = Start-Process -FilePath $CFturbo_Exe -ArgumentList "-batch `"$Current_CFT`"" -WorkingDirectory $WorkingDir -Wait -PassThru -NoNewWindow

if ($cftProcess.ExitCode -eq 2) {
    Write-Warning "CFturbo 几何生成失败 (ExitCode=2)，退出。"
    exit 1
} elseif ($cftProcess.ExitCode -eq 1) {
    Write-Warning "CFturbo 完成但有警告 (ExitCode=1)，继续执行..."
    # 不退出，继续后续步骤
}
if ($cftProcess.ExitCode -eq 2) {
    exit 1
}

# 不管 exit code 是 0 还是 1，验证关键输出文件
if (-not (Test-Path $Export_Hub) -or 
    -not (Test-Path $Export_Shroud) -or 
    -not (Test-Path $Export_Profile)) {
    Write-Warning "CFturbo 未生成 curve 文件，真正失败。"
    exit 1
}
Write-Host "CFturbo curve 文件验证通过，继续网格划分..."
# ===== CFturbo 运行完成后，解析 log 文件 =====
$cftLogPath = Join-Path $WorkingDir "run_cfturbo.log"
$cftResPath = Join-Path $WorkingDir "0908-2.cft-res"

# 定义致命警告关键词（出现任意一个即拦截）
$fatalKeywords = @(
    "choked flow",
    "Thermodynamic state cannot be calculated",
    "Suction diameter dS < choking diameter",
    "cm-calculation failed",
    "Grid generation or cm-calculation failed",
    "Calculation of LE blade angles not possible",
    "A reasonable thermodynamic state could not be calculated"
)
if (Test-Path $cftLogPath) {
    $logContent = Get-Content $cftLogPath -Raw
    $geometrySummary = Get-GeometrySummary -CftResPath $cftResPath -LogPath $cftLogPath
    $geometrySummary | ConvertTo-Json -Depth 6 | Set-Content -Path $Geometry_Summary -Encoding UTF8
    
    $fatalFound = @()
    foreach ($keyword in $fatalKeywords) {
        if ($logContent -match [regex]::Escape($keyword)) {
            $fatalFound += $keyword
        }
    }
    
    if ($fatalFound.Count -gt 0) {
        Write-Warning "!!! CFturbo 存在致命警告，该点物理上不可行，终止流水线："
        foreach ($f in $fatalFound) {
            Write-Warning "    - $f"
        }
        exit 2   # 用 exit code=2 表示硬失败（区别于 exit code=1 的 CFturbo 警告）
    } else {
        Write-Host ">>> CFturbo log 检查通过，无致命警告，继续网格划分。" -ForegroundColor Green
    }
} else {
    Write-Warning "未找到 CFturbo log 文件，跳过警告检查。"
    $geometrySummary = Get-GeometrySummary -CftResPath $cftResPath -LogPath $cftLogPath
    $geometrySummary | ConvertTo-Json -Depth 6 | Set-Content -Path $Geometry_Summary -Encoding UTF8
}

if ($geometrySummary.parsed) {
    Write-Host (">>> 几何摘要: beta1差={0:N2} deg | overlap最小因子={1:N3} | sweepWarn={2} | overlapWarn={3}" -f `
        $geometrySummary.beta1_diff_deg, $geometrySummary.overlap_factor_min, `
        $geometrySummary.warn_sweep, $geometrySummary.warn_overlap) -ForegroundColor Yellow
} else {
    Write-Warning "几何摘要未成功解析，将仅保留原始 log 与 .cft-res 文件。"
}
# ==============================================================================
# [3/3] 运行 TurboGrid 划分网格并强制导出
# ==============================================================================
Write-Host "--> [3/3] 准备运行 TurboGrid..."

# 1. 你的完美参数注入逻辑（生成专属的 .tgs 状态文件）
$tgsContent = Get-Content -Path $TGS_Template -Raw
$nBl_Int = [math]::Round($nBl)
$periodicAngle  = 360.0 / $nBl_Int
$tgsContent = $tgsContent -replace "\{BLADE_COUNT\}", $nBl_Int.ToString()
$tgsContent = $tgsContent -replace "\{HUB_CURVE\}", ($Export_Hub -replace "\\", "/")
$tgsContent = $tgsContent -replace "\{SHROUD_CURVE\}", ($Export_Shroud -replace "\\", "/")
$tgsContent = $tgsContent -replace "\{PROFILE_CURVE\}", ($Export_Profile -replace "\\", "/")
$tgsContent = $tgsContent -replace "\{OUTPUT_MESH\}", ($Export_Mesh -replace "\\", "/")
$tgsContent = $tgsContent -replace "\{PERIODIC_ANGLE\}", $periodicAngle.ToString("F5")
$tgsContent = $tgsContent -replace "\{TIP_CLEARANCE\}",  $TipClear.ToString("F6")
Set-Content -Path $Current_TGS -Value $tgsContent

# 2. 动态生成真正执行动作的宏文件 (.tse 发令枪)
$Current_TSE = Join-Path $WorkingDir "run_turbogrid.tse"

# 路径转正斜杠 (TurboGrid 强依赖)
$tgs_path_tg = $Current_TGS -replace "\\", "/"
$gtm_path_tg = $Export_Mesh -replace "\\", "/"

# 写入执行动作：加载状态 -> 录制的动作 -> 退出
$tseContent = @"
> um object=/TOPOLOGY SET, mode=normal, update=off
>readstate filename=$tgs_path_tg, mode = \
append, load = false
> update
>savemesh filename=$gtm_path_tg, coorddata=Off, \
onedomain=true, single=Off, units=m, solver=cfx5
> update
>quit
"@
Set-Content -Path $Current_TSE -Value $tseContent

# 3. 运行 TurboGrid，这次喂给它的是带有动作指令的 .tse 宏！
$tgProcess = Start-Process -FilePath $TurboGrid_Exe -ArgumentList "-batch `"$Current_TSE`"" -WorkingDirectory $WorkingDir -Wait -PassThru -NoNewWindow

# 4. 检查是否成功生成了网格文件
if (Test-Path $Export_Mesh) {
    Write-Host ">>> [成功] 网格文件已生成: $Export_Mesh" -ForegroundColor Green
    exit 0
} else {
    Write-Warning "!!! TurboGrid 未输出 .gtm 文件，网格映射失败。"
    exit 2
}
} catch {
    Write-Error "脚本执行中发生意外错误: $_"
    # [新增] 打印出究竟是哪一行代码引发的报错！
    Write-Error "具体报错行数: $($_.InvocationInfo.PositionMessage)"
    Write-Error "堆栈跟踪: $($_.ScriptStackTrace)"
    exit 1
}
