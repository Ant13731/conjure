//
//  HostListView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-17.
//
import ARKit
import AVFoundation
import SwiftUI

struct HostListView: View {
    @EnvironmentObject var hostListSettings: PersistentSettings<HostListSettings>
    @State private var editingHost: HostSettings?
    @State private var showAddHost = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading) {
                ForEach(hostListSettings.value.hosts) { host in
                    entry(host: host)
                }
                Divider()
                addHostEntry
            }
        }
        .sheet(item: $editingHost) { host in
            EditHostView(hostToEdit: host)
                .presentationDetents([.fraction(0.75)])
        }
        .sheet(isPresented: $showAddHost) {
            EditHostView()
                .presentationDetents([.fraction(0.75)])
        }
        .padding(.bottom, 24)
        .navigationTitle("Hosts")
    }

    var addHostEntry: some View {
        Button {
            showAddHost = true
        } label: {
            HStack {
                Image(systemName: "plus.circle.fill")
                Text("Add Host")
                Spacer()
            }
            .padding()
            .font(.headline)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)
            )
        }
        .padding()
    }

    @ViewBuilder
    private func entry(host: HostSettings) -> some View {
        HStack {
            Button {
                hostListSettings.value.currentHost = host
            } label: {
                HStack {
                    Text(host.friendlyName ?? host.ipAddress)
                        .font(.body)
                        .padding()
                    Spacer()
                }
            }
            .buttonStyle(.borderless)

            HStack(alignment: .center, spacing: 8) {
                if hostListSettings.value.currentHost == host {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                Button {
                    deleteHost(host)
                } label: {
                    Image(systemName: "trash")
                        .buttonStyle(.borderless)
                        .foregroundStyle(.red)
                }
                Button {
                    editingHost = host
                } label: {
                    Image(systemName: "pencil")
                        .buttonStyle(.borderless)
                        .foregroundStyle(.white)
                }

            }.padding()

        }
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.ultraThinMaterial)
        )
        .padding(.horizontal)
    }

    private func deleteHost(_ host: HostSettings) {
        if let index = hostListSettings.value.hosts.firstIndex(of: host) {
            hostListSettings.value.hosts.remove(at: index)
        }
    }

}

struct EditHostView: View {
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var hostListSettings: PersistentSettings<HostListSettings>

    var hostToEdit: HostSettings?

    @State private var friendlyName: String = ""
    @State private var ipAddress: String = ""
    @State private var port: String = ""

    var body: some View {
        Form {
            VStack {
                Text(hostToEdit == nil ? "Add Host" : "Edit Host")
                    .font(.headline)
                    .padding()
                TextField("Friendly Name", text: $friendlyName)
                TextField("IP Address", text: $ipAddress)
                    .keyboardType(.numbersAndPunctuation)
                TextField("Port", text: $port)
                    .keyboardType(.numbersAndPunctuation)
                Spacer()
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark").font(.system(size: 22))
                            .padding()
                            .foregroundStyle(.white)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                            .shadow(radius: 4)
                    }
                    Spacer()
                    Button {
                        saveHost()
                        dismiss()
                    } label: {
                        Image(systemName: "checkmark").font(.system(size: 22))
                            .padding()
                            .foregroundStyle(.white)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                            .shadow(radius: 4)
                    }
                }
            }
        }
        .onAppear {
            if let host = hostToEdit {
                friendlyName = host.friendlyName ?? ""
                ipAddress = host.ipAddress
                port = host.port
            }
        }
    }
    private func saveHost() {
        if var hostToEdit = hostToEdit {
            hostToEdit.friendlyName = friendlyName.isEmpty ? nil : friendlyName
            hostToEdit.ipAddress = ipAddress
            hostToEdit.port = port

            if let index = hostListSettings.value.hosts.firstIndex(where: { $0.id == hostToEdit.id }
            ) {
                hostListSettings.value.hosts[index] = hostToEdit

                // If new host was previously the current host, update that as well
                if hostListSettings.value.currentHost?.id == hostToEdit.id {
                    hostListSettings.value.currentHost = hostToEdit
                }

                return
            }
            print("Failed to find host to edit in store, adding as new host instead.")

        }

        let newHost = HostSettings(
            ipAddress: ipAddress,
            port: port,
            friendlyName: friendlyName.isEmpty ? nil : friendlyName,
        )

        hostListSettings.value.hosts.append(newHost)
    }
}
