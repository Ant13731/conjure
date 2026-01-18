//
//  VideoStreamView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-17.
//
import ARKit
import AVFoundation
import SwiftUI
import WebRTC

struct VideoStreamView: View {
    var body: some View {
        Color.black
            .overlay(
                Text("Video Stream")
                    .foregroundColor(.white)
            )
    }
}

struct HostListView: View {
    var body: some View {
        Color.black
            .overlay(
                Text("Hosts List")
                    .foregroundColor(.white)
            )
    }
}
