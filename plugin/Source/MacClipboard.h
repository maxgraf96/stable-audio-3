#pragma once

namespace sa3plugin {

// Put `posixPath` on the macOS general pasteboard as a file-URL so a
// subsequent ⌘V in Finder, Ableton Live, Logic, etc. lands the file at
// the cursor. Returns true on success. macOS-only — implementation lives
// in MacClipboard.mm.
bool copyFileToClipboard(const char* posixPath);

// Spawn `helperBinary <posixPath>` with the macOS responsibility chain
// disclaimed, so the helper isn't attributed back to the calling process
// (Live, Logic, etc). This is what's needed for Live's ⌘V handler to
// honour the resulting pasteboard entry — when Live spawns a normal
// subprocess via std::system / posix_spawn, the child's responsible
// process stays as Live, and Live filters out clipboards "owned by
// itself". Returns true if the spawn succeeded (the helper's own write
// happens asynchronously inside the child).
bool spawnDisclaimedClipboardWriter(const char* helperBinary,
                                    const char* posixPath);

// Force the host app (Live, Logic, …) to invalidate its cached
// pasteboard state. Live's ⌘V doesn't re-read NSPasteboard on every
// paste — it caches at applicationDidBecomeActive: time. After we write
// from a disclaimed subprocess Live has no idea the pasteboard moved;
// toggling NSApp's active state is what Finder→Live workflow effectively
// does for free (Finder paste deactivates Live momentarily). This call
// runs synchronously on the AppKit main thread.
void refreshHostPasteboardCache();

}  // namespace sa3plugin
