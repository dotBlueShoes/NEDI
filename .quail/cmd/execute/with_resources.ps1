param (
	[string]$project_dir = "",
	[string]$project_name = ""
)

# Because sources are not connected to the executable
#  but placed as a linkfile inside build directory...
#  When running from console we simply need to change the path to build's path.

Push-Location $project_dir
& ./$project_name.exe $args

if ( -not $? ) {
    Write-Host "EXE_WITH_RES.PS1 -> Execution returned a failure as STATUS!" -ForegroundColor Red
}

Pop-Location
