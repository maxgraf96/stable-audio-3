; SA3 Variations — Windows installer.
;
; Build with:
;   plugin\scripts\build_installer.ps1
; which stages the payload into plugin\build\installer_payload\ first, then
; runs ISCC over this script.
;
; Deliberately a per-user install (no admin, no UAC prompt): everything lands
; under %LOCALAPPDATA% so a colleague without local admin — common on managed
; work machines — can still install it.
;
; The installer stays ~20 MB. The two large payloads are fetched on the target
; machine instead of shipped:
;   - the Python runtime (~5.6 GB installed) via scripts\setup_runtime.ps1
;   - the model weights   (~5.5 GB)          via scripts\install_models.ps1
; That is partly a size decision and partly a hard constraint: GitHub caps
; release assets at 2 GB.

#define AppName        "SA3 Variations"
#define AppPublisher   "MaxGraf"
#define AppExeName     "SA3 Variations.exe"
#ifndef AppVersion
  #define AppVersion   "0.1.0"
#endif
#ifndef PayloadDir
  #define PayloadDir   "..\build\installer_payload"
#endif
#ifndef OutputDir
  #define OutputDir    "..\build\release_assets"
#endif

[Setup]
AppId={{8F3C5A21-6D4E-4B92-9C17-5A0E2D7B4F63}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user: no elevation, so no admin rights needed.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=SA3-Variations-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The app is x64-only: torch cu128 has no 32-bit build.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
DisableDirPage=no
DirExistsWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full";   Description: "Full install (recommended)"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "app";     Description: "SA3 Variations application"; Types: full custom; Flags: fixed
Name: "runtime"; Description: "Python runtime — downloads ~2.5 GB during install"; Types: full
Name: "models";  Description: "Model weights — downloads ~5.5 GB during install"; Types: full

[Files]
Source: "{#PayloadDir}\{#AppExeName}"; DestDir: "{app}";          Components: app; Flags: ignoreversion
Source: "{#PayloadDir}\app\*";         DestDir: "{app}\app";      Components: app; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\scripts\*";     DestDir: "{app}\scripts";  Components: app; Flags: ignoreversion recursesubdirs
Source: "{#PayloadDir}\tools\uv.exe";  DestDir: "{app}\tools";    Components: app; Flags: ignoreversion
Source: "{#PayloadDir}\README.txt";    DestDir: "{app}";          Components: app; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#AppName}";                     Filename: "{app}\{#AppExeName}"
Name: "{group}\Set up Python runtime";          Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\scripts\setup_runtime.ps1"""; Comment: "Re-run if the runtime setup failed or was skipped"
Name: "{group}\Download models";                Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\scripts\install_models.ps1"""; Comment: "Re-run if the model download failed or was skipped"
Name: "{group}\Worker log";                     Filename: "{localappdata}\{#AppName}\worker.log"
Name: "{userdesktop}\{#AppName}";               Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

; Both downloads run in a visible PowerShell window rather than silently:
; they take many minutes and move gigabytes, and a progress bar the user can
; watch (and read errors from) beats a frozen installer. Both are re-runnable
; from the Start Menu, so a failure here is recoverable without reinstalling.
[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\scripts\setup_runtime.ps1"""; StatusMsg: "Setting up the Python runtime (this downloads ~2.5 GB)..."; Components: runtime; Flags: waituntilterminated
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\scripts\install_models.ps1"""; StatusMsg: "Downloading model weights (~5.5 GB)..."; Components: models; Flags: waituntilterminated
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; Uninstall removes the program and its runtime, but NOT
; %LOCALAPPDATA%\SA3 Variations\models — forcing a 5.5 GB re-download on every
; reinstall would be hostile. Users who want the space back are told where to
; look, in README.txt and RELEASE_WINDOWS.md.
[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app\__pycache__"

[Code]
function InitializeSetup(): Boolean;
var
  FreeMB: Cardinal;
begin
  Result := True;
  // ~16 GB: runtime (~6) + models (~5.5) + working room. Warn rather than
  // block — a custom install of just the app needs almost nothing, and the
  // user may be pointing the install at a different drive.
  if GetSpaceOnDisk(ExpandConstant('{localappdata}'), True, FreeMB, FreeMB) then
  begin
    if FreeMB < 16384 then
      if MsgBox('This drive has about ' + IntToStr(FreeMB div 1024) +
                ' GB free. A full install needs roughly 16 GB (Python runtime plus model weights).'#13#10#13#10 +
                'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
        Result := False;
  end;
end;
