// MacClipboard.mm — file-URL → general NSPasteboard.
//
// We hand-craft the pasteboard layout to mirror Finder's ⌘C exactly:
// a single item carrying a file-REFERENCE URL (file:///.file/id=<inode>),
// plus the displayed filename as utf8 + utf16-external text. Diffing our
// pasteboard against Finder's showed that Live's paste handler rejects
// the plain path-URL form (file:///abs/path) — it only accepts the inode
// reference form. The legacy NSFilenamesPboardType isn't written by
// Finder either, so we drop it.
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#include <spawn.h>
#include <unistd.h>

#include "MacClipboard.h"

extern char** environ;

// Darwin-private declaration. Available in libSystem from macOS 10.14;
// Terminal.app, iTerm, etc. use it to disclaim parent responsibility for
// children they spawn so per-process Mach services (notably the
// pasteboard) attribute the child's actions correctly.
extern "C" int responsibility_spawnattrs_setdisclaim(posix_spawnattr_t attrs,
                                                     int disclaim);

namespace sa3plugin {

// Drive NSPasteboard from the AppKit main thread. Inside a VST3 host the
// native-function callback can land on a WebKit IPC thread, and off-main
// pasteboard writes occasionally drop on the floor.
bool copyFileToClipboard(const char* posixPath)
{
    if (posixPath == nullptr || posixPath[0] == '\0') return false;
    NSString* nsPath = [NSString stringWithUTF8String:posixPath];
    NSURL*    url    = [NSURL fileURLWithPath:nsPath];
    if (url == nil) return false;

    __block BOOL ok = NO;
    void (^writeBlock)(void) = ^{
        @autoreleasepool {
            NSPasteboard* pb     = [NSPasteboard generalPasteboard];
            NSInteger     before = [pb changeCount];
            NSURL*        refURL = [url fileReferenceURL] ?: url;
            NSString*     leaf   = [url lastPathComponent];
            [pb clearContents];
            BOOL urlOk = [pb writeObjects:@[refURL]];
            [pb setString:leaf forType:NSPasteboardTypeString];
            NSData* utf16 = [leaf dataUsingEncoding:NSUTF16LittleEndianStringEncoding];
            if (utf16 != nil) {
                [pb setData:utf16
                    forType:(NSPasteboardType)@"public.utf16-external-plain-text"];
            }
            NSInteger after = [pb changeCount];
            NSLog(@"[SA3] clipboard write path=%@ refURL=%@ urlOk=%d "
                  @"changeCount %ld → %ld",
                  nsPath, refURL, urlOk, (long)before, (long)after);
            ok = urlOk;
        }
    };
    if ([NSThread isMainThread]) writeBlock();
    else dispatch_sync(dispatch_get_main_queue(), writeBlock);
    return ok == YES;
}

bool spawnDisclaimedClipboardWriter(const char* helperBinary,
                                    const char* posixPath)
{
    if (helperBinary == nullptr || helperBinary[0] == '\0') return false;
    if (posixPath    == nullptr || posixPath[0]    == '\0') return false;

    posix_spawnattr_t attr;
    if (posix_spawnattr_init(&attr) != 0) return false;
    responsibility_spawnattrs_setdisclaim(&attr, 1);

    // Detach stdio so the child doesn't keep our pipes open.
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO,  "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    char* const argv[] = {
        const_cast<char*>(helperBinary),
        const_cast<char*>(posixPath),
        nullptr,
    };

    pid_t pid = 0;
    int rc = posix_spawn(&pid, helperBinary, &actions, &attr, argv, environ);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) {
        NSLog(@"[SA3] posix_spawn(%s) failed: %s", helperBinary, strerror(rc));
        return false;
    }
    NSLog(@"[SA3] spawned disclaimed helper pid=%d path=%s", pid, posixPath);
    return true;
}

void refreshHostPasteboardCache()
{
    // Toggling the host's active state forces it to fire
    // applicationDidBecomeActive: which is where most Cocoa apps refresh
    // any pasteboard contents they've cached. Quick enough to be
    // imperceptible in practice (~one runloop tick).
    void (^toggle)(void) = ^{
        [NSApp deactivate];
        [NSApp activateIgnoringOtherApps:YES];
    };
    if ([NSThread isMainThread]) toggle();
    else dispatch_sync(dispatch_get_main_queue(), toggle);
}

}  // namespace sa3plugin
