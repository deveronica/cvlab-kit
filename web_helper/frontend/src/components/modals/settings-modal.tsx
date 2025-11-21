import React from "react";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '../ui/dialog';
import { Button } from '../ui/button';
import { X } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="w-full max-w-md h-[90vh] overflow-y-auto flex flex-col gap-4" hideCloseButton>
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <DialogTitle>Settings</DialogTitle>
            <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <DialogDescription>
            Configure your application preferences
          </DialogDescription>
        </div>

        <div className="space-y-4">
          {/* Additional Settings Placeholder */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Experiment Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                Additional experiment configuration options will be available here.
              </div>
            </CardContent>
          </Card>
        </div>
      </DialogContent>
    </Dialog>
  );
}