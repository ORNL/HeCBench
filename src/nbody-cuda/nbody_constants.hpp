#pragma once

// Shared by the device step and the host reference. A mismatch here
// shows up as a verification failure.
constexpr float kSofteningSquared = 1e-3f;
// Keeps the force finite when two particles occupy the same position.
constexpr float kG = 6.67259e-11f;
