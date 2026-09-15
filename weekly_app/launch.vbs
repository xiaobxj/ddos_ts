Option Explicit
Dim shell, files, projectRoot, pythonPath, scriptPath
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
projectRoot = files.GetParentFolderName(files.GetParentFolderName(WScript.ScriptFullName))
pythonPath = files.BuildPath(projectRoot, "research_v4\.venv_gpu\Scripts\pythonw.exe")
scriptPath = files.BuildPath(projectRoot, "weekly_app\app.py")
If Not files.FileExists(pythonPath) Then
    MsgBox "Python environment not found: " & pythonPath, 16, "Weekly Research"
    WScript.Quit 1
End If
shell.Run Chr(34) & pythonPath & Chr(34) & " -B " & Chr(34) & scriptPath & Chr(34), 0, False
