-- Hammerspoon configuration for Smart Scribe
-- Add this to ~/.hammerspoon/init.lua

-- Adjust this to your actual python3 path (e.g. from `which python3`)
local python = "/opt/homebrew/bin/python3"
local script = os.getenv("HOME") .. "/Projects/smart-scribe/smart_scribe.py"

local scribeTask = nil

local function stopScribe(reason)
    if scribeTask and scribeTask:isRunning() then
        scribeTask:terminate()
        scribeTask = nil
        hs.notify.show("Scribe", "", reason or "Cancelled")
        hs.task.new("/usr/bin/afplay", nil, {"/System/Library/Sounds/Basso.aiff"}):start()
    end
end

local function startScribe(extraArgs)
    -- If already running, treat this as "cancel"
    if scribeTask and scribeTask:isRunning() then
        stopScribe("Cancelled")
        return
    end

    local args = { script }
    if extraArgs then
        for _, a in ipairs(extraArgs) do
            table.insert(args, a)
        end
    end

    scribeTask = hs.task.new(python, function(exitCode, stdOut, stdErr)
        scribeTask = nil
        if exitCode ~= 0 then
            local msg = stdErr
            if not msg or #msg == 0 then
                msg = "Exit code: " .. tostring(exitCode)
            end
            hs.notify.show("Scribe error", "", msg)
            print("Scribe error: " .. msg)
        end
    end, args)

    scribeTask:start()
end

-- Record → Transcribe → Paste
hs.hotkey.bind({ "ctrl", "alt" }, "space", function()
    startScribe(nil)
end)

-- Record → Transcribe → Clipboard only
hs.hotkey.bind({ "ctrl", "alt", "shift" }, "space", function()
    startScribe({ "--clipboard" })
end)
