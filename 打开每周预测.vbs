Option Explicit
Dim shell, files, launcher
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
launcher = files.BuildPath(files.GetParentFolderName(WScript.ScriptFullName), "weekly_app\launch.vbs")
shell.Run "wscript.exe " & Chr(34) & launcher & Chr(34), 0, False
