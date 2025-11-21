import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';

export function SettingsView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Settings</h1>
        <p className="text-muted-foreground">
          Configure your experiment platform
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Device Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Device Management</CardTitle>
            <CardDescription>
              Configure device monitoring and thresholds
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Heartbeat Interval</label>
              <select className="w-full p-2 border border-border rounded-lg bg-background">
                <option>3 seconds</option>
                <option>5 seconds</option>
                <option>10 seconds</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Disconnect Threshold</label>
              <select className="w-full p-2 border border-border rounded-lg bg-background">
                <option>30 seconds</option>
                <option>60 seconds</option>
                <option>2 minutes</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <input type="checkbox" id="auto-restart" className="rounded" defaultChecked />
              <label htmlFor="auto-restart" className="text-sm">Auto-restart failed jobs</label>
            </div>
          </CardContent>
        </Card>

        {/* Queue Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Queue Management</CardTitle>
            <CardDescription>
              Configure experiment queue behavior
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Queue Policy</label>
              <select className="w-full p-2 border border-border rounded-lg bg-background">
                <option>FIFO (First In, First Out)</option>
                <option>Priority-based</option>
                <option>Round Robin</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max Concurrent Jobs</label>
              <input
                type="number"
                defaultValue="1"
                min="1"
                max="10"
                className="w-full p-2 border border-border rounded-lg bg-background"
              />
            </div>

            <div className="flex items-center gap-2">
              <input type="checkbox" id="queue-notifications" className="rounded" defaultChecked />
              <label htmlFor="queue-notifications" className="text-sm">Enable queue notifications</label>
            </div>
          </CardContent>
        </Card>

        {/* Storage Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Storage & Logs</CardTitle>
            <CardDescription>
              Manage experiment data and logging
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Log Level</label>
              <select className="w-full p-2 border border-border rounded-lg bg-background">
                <option>INFO</option>
                <option>DEBUG</option>
                <option>WARNING</option>
                <option>ERROR</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Data Retention</label>
              <select className="w-full p-2 border border-border rounded-lg bg-background">
                <option>30 days</option>
                <option>90 days</option>
                <option>1 year</option>
                <option>Forever</option>
              </select>
            </div>

            <button className="w-full py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition-colors">
              🗑️ Clear Old Logs
            </button>
          </CardContent>
        </Card>

        {/* Notification Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
            <CardDescription>
              Configure when and how you receive updates
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Experiment Completion</p>
                <p className="text-sm text-muted-foreground">Get notified when experiments finish</p>
              </div>
              <input type="checkbox" className="rounded" defaultChecked />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Device Alerts</p>
                <p className="text-sm text-muted-foreground">Notify on device disconnections</p>
              </div>
              <input type="checkbox" className="rounded" defaultChecked />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Queue Updates</p>
                <p className="text-sm text-muted-foreground">Updates on queue status changes</p>
              </div>
              <input type="checkbox" className="rounded" />
            </div>
          </CardContent>
        </Card>

        {/* System Info */}
        <Card>
          <CardHeader>
            <CardTitle>System Information</CardTitle>
            <CardDescription>
              Platform version and status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Platform Version:</span>
              <span>v1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Backend Status:</span>
              <span className="text-green-600">Online</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Database:</span>
              <span>Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Last Backup:</span>
              <span>2 hours ago</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}