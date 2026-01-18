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
    @EnvironmentObject var connectionConfigStore: ConnectionConfigStore
    @State private var editingHost: HostConfig?
    @State private var showAddHost = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading) {
                ForEach(connectionConfigStore.hosts) { host in
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
    private func entry(host: HostConfig) -> some View {
        HStack {
            Button {
                connectionConfigStore.currentHostConfig = host
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
                if connectionConfigStore.currentHostConfig == host {
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

    private func deleteHost(_ host: HostConfig) {
        if let index = connectionConfigStore.hosts.firstIndex(of: host) {
            connectionConfigStore.hosts.remove(at: index)
        }
    }

}

struct EditHostView: View {
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var connectionConfigStore: ConnectionConfigStore

    var hostToEdit: HostConfig?

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

            if let index = connectionConfigStore.hosts.firstIndex(where: { $0.id == hostToEdit.id })
            {
                connectionConfigStore.hosts[index] = hostToEdit
                return
            }
            print("Failed to find host to edit in store, adding as new host instead.")

        }

        let newHost = HostConfig(
            ipAddress: ipAddress,
            port: port,
            friendlyName: friendlyName.isEmpty ? nil : friendlyName,
        )

        // if let editing = hostToEdit,
        //     let index = connectionConfigStore.hosts.firstIndex(where: { $0.id == hostToEdit.id })
        // {
        //     connectionConfigStore.hosts[index] = newHost
        // } else {
        connectionConfigStore.hosts.append(newHost)
        // }
    }
}
