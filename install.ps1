<#
.SYNOPSIS
    Sets up dc-sub-fixer on Windows: virtual environment, GPU wheels, Hi-SAM
    source and model checkpoints.

.DESCRIPTION
    The awkward parts of this install are not obvious, so they are encoded here
    rather than left to a reader of requirements.txt:

      * torch and paddlepaddle-gpu come from their own package indexes, and the
        right channel depends on the CUDA version the driver supports. The
        script reads that from nvidia-smi and picks, and both are overridable.
      * Paddle publishes no cp312 Windows wheel on its cu132 channel, so the
        CUDA 13 choice is cu130 even when torch is on cu132. The two coexist.
      * paddleocr depends on opencv-contrib-python. Installing plain
        opencv-python beside it puts two builds of cv2 in one site-packages
        and they overwrite each other's files, leaving a broken cv2 behind.
      * Hi-SAM is cloned, not vendored, and needs the base SAM backbone
        checkpoint in addition to its own.

    Verification at the end runs a convolution through Paddle, because a wrong
    cuDNN DLL search path only shows up once a conv is attempted, long after
    `import paddle` has appeared to succeed.

.PARAMETER Python
    Interpreter to build the venv from. Default: the newest 3.12 the launcher
    knows about, else whatever `python` resolves to.

.PARAMETER TorchIndex
    Override the torch wheel index, e.g. https://download.pytorch.org/whl/cu126

.PARAMETER PaddleIndex
    Override the paddle wheel index, e.g.
    https://www.paddlepaddle.org.cn/packages/stable/cu126/

.PARAMETER SkipModels
    Do not download the base SAM checkpoint (1.2 GB).

.PARAMETER SkipVerify
    Skip the post-install checks.

.PARAMETER Force
    Delete and rebuild an existing .venv.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -TorchIndex https://download.pytorch.org/whl/cu126 -PaddleIndex https://www.paddlepaddle.org.cn/packages/stable/cu126/
#>

[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$TorchIndex = "",
    [string]$PaddleIndex = "",
    [switch]$SkipModels,
    [switch]$SkipVerify,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$ModelsDir = Join-Path $Root "models"
$HiSamDir = Join-Path $Root "third_party\Hi-SAM"

$script:Warnings = @()

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Ok($text) { Write-Host "  [ok] $text" -ForegroundColor Green }
function Info($text) { Write-Host "  $text" -ForegroundColor Gray }
function Warn($text) {
    Write-Host "  [!] $text" -ForegroundColor Yellow
    $script:Warnings += $text
}
function Die($text) {
    Write-Host "`n  [x] $text" -ForegroundColor Red
    exit 1
}

function Invoke-Pip {
    param([string[]]$Arguments, [string]$What)
    Info "pip install $($Arguments -join ' ')"
    & $VenvPy -m pip install --no-input --disable-pip-version-check @Arguments
    if ($LASTEXITCODE -ne 0) { Die "failed to install $What (pip exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- interpreter
Step "Python"
if (-not $Python) {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += "py -3.12"
        $candidates += "py -3"
    }
    $candidates += "python"
    foreach ($c in $candidates) {
        $parts = $c.Split(" ")
        try {
            $v = & $parts[0] $parts[1..($parts.Length - 1)] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { $v = $null }
        if ($v) { $Python = $c; break }
    }
}
if (-not $Python) { Die "no Python interpreter found. Install Python 3.12 from python.org." }

$pyParts = $Python.Split(" ")
$pyExe = $pyParts[0]
$pyArgs = @()
if ($pyParts.Length -gt 1) { $pyArgs = $pyParts[1..($pyParts.Length - 1)] }
$pyVer = & $pyExe @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
Info "using $Python (Python $pyVer)"
$major, $minor = $pyVer.Split(".")
if ([int]$major -ne 3 -or [int]$minor -lt 10) { Die "Python 3.10 or newer is required, found $pyVer" }
if ([int]$minor -ne 12) { Warn "developed and tested on Python 3.12; $pyVer may resolve different wheels" }

# ---------------------------------------------------------------------- GPU
Step "GPU"
$cudaMajor = 0
$cudaMinor = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($smi) { Info "device: $smi" }
    # Recent Windows drivers label this "CUDA UMD Version"; older ones and
    # Linux say "CUDA Version". Missing it silently picks wheels for the wrong
    # CUDA, so accept both and say so when neither matches.
    $header = (& nvidia-smi) -join "`n"
    $m = [regex]::Match($header, "CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)")
    if ($m.Success) {
        $cudaMajor = [int]$m.Groups[1].Value
        $cudaMinor = [int]$m.Groups[2].Value
        Info "driver supports CUDA up to $cudaMajor.$cudaMinor"
    } else {
        Warn "could not read the CUDA version from nvidia-smi; assuming 12.6. Pass -TorchIndex and -PaddleIndex if that is wrong."
    }
} else {
    Warn "nvidia-smi not found. A CUDA GPU is required; falling back to CUDA 12.6 wheels."
}

$cudaNum = $cudaMajor * 100 + $cudaMinor
if (-not $TorchIndex) {
    # Highest torch channel the driver can run.
    $TorchIndex = "https://download.pytorch.org/whl/cu126"
    if ($cudaNum -ge 1208) { $TorchIndex = "https://download.pytorch.org/whl/cu128" }
    if ($cudaNum -ge 1300) { $TorchIndex = "https://download.pytorch.org/whl/cu130" }
    if ($cudaNum -ge 1302) { $TorchIndex = "https://download.pytorch.org/whl/cu132" }
}
if (-not $PaddleIndex) {
    # Paddle ships no cp312 Windows wheel on cu132, so CUDA 13 stops at cu130.
    $PaddleIndex = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    if ($cudaNum -ge 1209) { $PaddleIndex = "https://www.paddlepaddle.org.cn/packages/stable/cu129/" }
    if ($cudaNum -ge 1300) { $PaddleIndex = "https://www.paddlepaddle.org.cn/packages/stable/cu130/" }
}
Info "torch index : $TorchIndex"
Info "paddle index: $PaddleIndex"

# --------------------------------------------------------------------- venv
Step "Virtual environment"
if ((Test-Path $VenvDir) -and $Force) {
    Info "removing existing .venv (-Force)"
    Remove-Item $VenvDir -Recurse -Force
}
if (Test-Path $VenvPy) {
    Ok ".venv already exists (use -Force to rebuild)"
} else {
    & $pyExe @pyArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Die "could not create the virtual environment" }
    Ok "created $VenvDir"
}
& $VenvPy -m pip install --quiet --upgrade --disable-pip-version-check pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Die "could not update pip" }
Ok "pip up to date"

# ----------------------------------------------------------------- packages
Step "PyTorch (GPU)"
Invoke-Pip @("torch", "torchvision", "--index-url", $TorchIndex) "torch"

Step "PaddlePaddle (GPU)"
Invoke-Pip @("paddlepaddle-gpu", "-i", $PaddleIndex) "paddlepaddle-gpu"

Step "Remaining dependencies"
Invoke-Pip @("-r", (Join-Path $Root "requirements.txt")) "requirements.txt"

# paddleocr pulls opencv-contrib-python; a second cv2 build breaks both.
# Read the list as JSON rather than calling `pip show`: that writes a warning
# to stderr when the package is absent, which PowerShell turns into a
# terminating error under ErrorActionPreference = Stop.
$installed = (& $VenvPy -m pip list --format=json --disable-pip-version-check) | ConvertFrom-Json
$hasPlain = $installed | Where-Object { $_.name -eq "opencv-python" }
if ($hasPlain) {
    Warn "opencv-python was installed alongside opencv-contrib-python; removing it"
    & $VenvPy -m pip uninstall -y opencv-python | Out-Null
    # Uninstalling one deletes files the other shares, so put contrib back whole.
    & $VenvPy -m pip install --force-reinstall --no-deps opencv-contrib-python | Out-Null
}

# ------------------------------------------------------------------- Hi-SAM
Step "Hi-SAM source"
if (Test-Path (Join-Path $HiSamDir "hi_sam")) {
    Ok "already present at third_party\Hi-SAM"
} elseif (Get-Command git -ErrorAction SilentlyContinue) {
    & git clone --depth 1 https://github.com/ymy-k/Hi-SAM.git $HiSamDir
    if ($LASTEXITCODE -ne 0) { Die "git clone of Hi-SAM failed" }
    Ok "cloned Hi-SAM"
} else {
    Die "git not found. Install Git, or clone manually:`n      git clone https://github.com/ymy-k/Hi-SAM.git third_party\Hi-SAM"
}

# ------------------------------------------------------------------- models
function Get-DriveUri {
    <#
        Google Drive will not hand over a large file from a /uc share link: it
        answers with an HTML "can't scan this for viruses" page instead, which
        saves happily as a .pth and only fails later, at load time, looking
        like a corrupt checkpoint. The usercontent endpoint with confirm=t
        returns the bytes directly.
    #>
    param([string]$FileId)
    return "https://drive.usercontent.google.com/download?id=$FileId&export=download&confirm=t"
}

function Save-Checkpoint {
    <#
        Download to a .part file and only put it in place once the size and
        hash check out. A truncated download left under the real name would be
        reported as "present" by the next run and then fail somewhere far less
        obvious.
    #>
    param(
        [string]$Uri,
        [string]$Destination,
        [long]$ExpectedSize,
        [string]$ExpectedSha256,
        [string]$Label
    )
    $part = "$Destination.part"
    if (Test-Path $part) { Remove-Item $part -Force }
    Info "downloading $Label ($([math]::Round($ExpectedSize / 1MB)) MB)"
    try {
        $old = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $Uri -OutFile $part -UseBasicParsing -TimeoutSec 1800
        $ProgressPreference = $old
    } catch {
        if (Test-Path $part) { Remove-Item $part -Force }
        Warn "could not download $Label`: $($_.Exception.Message)"
        return $false
    }

    $size = (Get-Item $part).Length
    if ($size -ne $ExpectedSize) {
        Remove-Item $part -Force
        # A few KB means the host served an interstitial or an error page.
        Warn "$Label is $size bytes, expected $ExpectedSize - the download was intercepted or truncated"
        return $false
    }
    $hash = (Get-FileHash $part -Algorithm SHA256).Hash
    if ($hash -ne $ExpectedSha256) {
        Remove-Item $part -Force
        Warn "$Label failed its checksum (got $hash) - the mirror is serving something else"
        return $false
    }
    Move-Item $part $Destination -Force
    Ok "downloaded and verified $Label"
    return $true
}

Step "Model checkpoints"
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

$checkpoints = @(
    @{
        Name = "sam_vit_l_0b3195.pth"
        Label = "base SAM ViT-L backbone"
        Uri = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        Size = 1249524607L
        Sha256 = "3ADCC4315B642A4D2101128F611684E8734C41232A17C648ED1693702A49A622"
        Manual = "https://github.com/facebookresearch/segment-anything#model-checkpoints"
    },
    @{
        Name = "sam_tss_l_textseg.pth"
        Label = "Hi-SAM text stroke segmentation (ViT-L, TextSeg)"
        Uri = (Get-DriveUri "1vEWn3fmlFnVPRyFyPWhRhp_1Upzd3eaM")
        Size = 122756163L
        Sha256 = "1A7399FD5B031383A3776B4375332D23B952BE616A735B545B3ABB7EB89D063F"
        Manual = "https://github.com/ymy-k/Hi-SAM"
    }
)

foreach ($ck in $checkpoints) {
    $path = Join-Path $ModelsDir $ck.Name
    if (Test-Path $path) {
        Ok "$($ck.Name) present"
        continue
    }
    if ($SkipModels) {
        Warn "$($ck.Name) missing (-SkipModels was given)"
        continue
    }
    $got = Save-Checkpoint -Uri $ck.Uri -Destination $path -ExpectedSize $ck.Size `
        -ExpectedSha256 $ck.Sha256 -Label $ck.Label
    if (-not $got) {
        Info "    fetch it manually from $($ck.Manual)"
        Info "    and save it into: $ModelsDir"
    }
}

# ------------------------------------------------------------------- verify
if (-not $SkipVerify) {
    Step "Verification"
    $check = Join-Path $env:TEMP "dcsubfixer_verify.py"
    @'
import sys
sys.path.insert(0, r"__ROOT__")
fail = []

try:
    import torch
    if torch.cuda.is_available():
        print("  [ok] torch %s, CUDA on %s" % (torch.__version__, torch.cuda.get_device_name(0)))
    else:
        fail.append("torch cannot see a CUDA device")
except Exception as exc:
    fail.append("torch: %r" % (exc,))

try:
    import cv2
    print("  [ok] opencv %s" % cv2.__version__)
except Exception as exc:
    fail.append("cv2: %r" % (exc,))

# Paddle has to be checked in its own process: torch is already imported here,
# and the two cannot share one (see dcsubfixer/_paddle_env.py). That is exactly
# how the pipeline runs detection, so this also proves the arrangement works.
# The convolution matters because a wrong cuDNN DLL path only surfaces there,
# well after `import paddle` has appeared to succeed.
import subprocess
paddle_probe = "\n".join([
    "from dcsubfixer import _paddle_env",
    "_paddle_env.apply()",
    "_paddle_env.isolate()",
    "import paddle",
    "paddle.nn.Conv2D(3, 8, 3)(paddle.randn([1, 3, 32, 32]))",
    "print('PADDLE_OK', paddle.__version__)",
])
try:
    proc = subprocess.run([sys.executable, "-c", paddle_probe], cwd=r"__ROOT__",
                          capture_output=True, text=True, timeout=300)
    line = [l for l in proc.stdout.splitlines() if l.startswith("PADDLE_OK")]
    if proc.returncode == 0 and line:
        print("  [ok] paddle %s in a subprocess, cuDNN convolution succeeded"
              % line[0].split()[1])
    else:
        tail = (proc.stderr.strip().splitlines() or ["exit %d" % proc.returncode])[-1]
        fail.append("paddle convolution: %s" % tail)
except Exception as exc:
    fail.append("paddle probe: %r" % (exc,))

try:
    import av
    from dcsubfixer import hisam
    print("  [ok] dcsubfixer imports, PyAV %s" % av.__version__)
except Exception as exc:
    fail.append("dcsubfixer: %r" % (exc,))

try:
    from PySide6 import QtWidgets  # noqa: F401
    print("  [ok] PySide6 available (GUI)")
except Exception as exc:
    print("  [!] PySide6 missing, GUI unavailable: %r" % (exc,))

if fail:
    print("\nFAILURES:")
    for f in fail:
        print("  - %s" % f)
    sys.exit(1)
sys.exit(0)
'@.Replace("__ROOT__", $Root) | Set-Content -Path $check -Encoding utf8

    & $VenvPy $check
    $verifyCode = $LASTEXITCODE
    Remove-Item $check -ErrorAction SilentlyContinue
    if ($verifyCode -ne 0) {
        Warn "verification reported problems (see above)"
    } else {
        Ok "all checks passed"
    }
}

# ------------------------------------------------------------------ summary
Step "Done"
if ($script:Warnings.Count -gt 0) {
    Write-Host "  Finished with $($script:Warnings.Count) warning(s):" -ForegroundColor Yellow
    foreach ($w in $script:Warnings) { Write-Host "    - $w" -ForegroundColor Yellow }
    Write-Host ""
}
Write-Host "  Tuning window :  dc-sub-fixer.bat" -ForegroundColor White
Write-Host "  Command line  :  .venv\Scripts\python -m dcsubfixer --help" -ForegroundColor White
Write-Host "  Tests         :  .venv\Scripts\python -m pytest tests" -ForegroundColor White
Write-Host ""
