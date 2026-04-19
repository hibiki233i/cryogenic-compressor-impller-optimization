$ErrorActionPreference = "Stop"

$repo = "hibiki233i/cryogenic-compressor-impller-optimization"
$branch = "codex/sync-local-workspace-20260419"
$baseBranch = "main"
$gh = "C:\Program Files\GitHub CLI\gh.exe"

function GhJson {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    $out = & $gh @Args 2>$null
    $text = ($out -join "`n").Trim()
    if (-not $text) {
        throw "Empty response from gh api: $($Args -join ' ')"
    }
    try {
        return $text | ConvertFrom-Json
    } catch {
        $match = [regex]::Match($text, '(\{.*\}|\[.*\])', [System.Text.RegularExpressions.RegexOptions]::Singleline)
        if (-not $match.Success) {
            throw "Non-JSON response from gh api: $text"
        }
        try {
            return $match.Value | ConvertFrom-Json
        } catch {
            throw "Failed to parse gh api response as JSON for '$($Args -join ' ')': $text"
        }
    }
}

$baseRef = GhJson @("api", "repos/$repo/git/ref/heads/$baseBranch")
$baseCommitSha = $baseRef.object.sha
$baseCommit = GhJson @("api", "repos/$repo/git/commits/$baseCommitSha")
$baseTreeSha = $baseCommit.tree.sha

$existingBranch = $null
try {
    $existingBranch = GhJson @("api", "repos/$repo/git/ref/heads/$branch")
} catch {
    $existingBranch = $null
}

$files = Get-ChildItem -Recurse -File |
    Where-Object {
        $_.FullName -notmatch "\\build\\" -and
        $_.FullName -notmatch "\\dist\\" -and
        $_.FullName -notmatch "\\__pycache__\\" -and
        $_.FullName -notmatch "\\.git\\"
    } |
    Sort-Object FullName

$treeElements = @()
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($PWD.Path.Length + 1).Replace("\", "/")
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $base64 = [Convert]::ToBase64String($bytes)
    $blob = GhJson @("api", "repos/$repo/git/blobs", "-X", "POST", "-f", "encoding=base64", "-f", "content=$base64")
    $treeElements += [ordered]@{
        path = $relativePath
        mode = "100644"
        type = "blob"
        sha  = $blob.sha
    }
}

$treePayload = @{
    base_tree = $baseTreeSha
    tree = $treeElements
} | ConvertTo-Json -Depth 6 -Compress
$tree = $treePayload | & $gh api "repos/$repo/git/trees" --input - 2>$null
$treeObj = (($tree -join "`n").Trim() | ConvertFrom-Json)

$commitPayload = @{
    message = "Sync local workspace excluding build artifacts"
    tree = $treeObj.sha
    parents = @($baseCommitSha)
} | ConvertTo-Json -Compress
$commit = $commitPayload | & $gh api "repos/$repo/git/commits" --input - 2>$null
$commitObj = (($commit -join "`n").Trim() | ConvertFrom-Json)

if ($existingBranch) {
    & $gh api "repos/$repo/git/refs/heads/$branch" -X PATCH -f "sha=$($commitObj.sha)" -f "force=true" | Out-Null
} else {
    & $gh api "repos/$repo/git/refs" -X POST -f "ref=refs/heads/$branch" -f "sha=$($commitObj.sha)" | Out-Null
}

$prTitle = "[codex] Sync local workspace excluding build artifacts"
$prBody = @"
This PR syncs the current local workspace into the repository while excluding generated build outputs.

What changed:
- updates the desktop app source and packaging files from the local workspace
- includes the recent GUI title/version text changes
- includes the PyInstaller entry/import fixes
- excludes `build/`, `dist/`, and `__pycache__/`

Validation:
- no runtime validation was executed in this environment because the local shell does not have a usable Python/git toolchain in PATH
"@

$existingPr = $null
try {
    $existingPr = GhJson @("api", "repos/$repo/pulls?state=open&head=hibiki233i:$branch&base=$baseBranch")
} catch {
    $existingPr = $null
}

if ($existingPr -and $existingPr.Count -gt 0) {
    $prUrl = $existingPr[0].html_url
} else {
    $prPayload = @{
        title = $prTitle
        head = $branch
        base = $baseBranch
        body = $prBody
        draft = $true
    } | ConvertTo-Json -Compress
    $pr = $prPayload | & $gh api "repos/$repo/pulls" --input - 2>$null
    $prObj = (($pr -join "`n").Trim() | ConvertFrom-Json)
    $prUrl = $prObj.html_url
}

[pscustomobject]@{
    branch = $branch
    files = $files.Count
    commit = $commitObj.sha
    pr = $prUrl
} | ConvertTo-Json -Compress
