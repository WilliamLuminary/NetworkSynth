import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    visible: true
    width: 720
    height: 780
    title: "NetworkSynth — Synthesis"

    // `controller` is exposed from Python as a context property.

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            Label {
                text: "Synthetic network generation"
                font.pixelSize: 16
                font.bold: true
                Layout.fillWidth: true
            }
            Label {
                text: "Mode"
                opacity: 0.6
            }
            ComboBox {
                id: modeBox
                model: controller.modeLabels
                currentIndex: controller.modes.indexOf(controller.mode)
                onActivated: controller.selectMode(currentIndex)
                enabled: !controller.running
                Layout.preferredWidth: 170
            }
        }
    }

    FileDialog {
        id: edgeDialog
        title: "Select edge-list CSV"
        nameFilters: ["CSV files (*.csv)", "All files (*)"]
        onAccepted: controller.setEdgeList(selectedFile)
    }
    FileDialog {
        id: positionsDialog
        title: "Select positions CSV"
        nameFilters: ["CSV files (*.csv)", "All files (*)"]
        onAccepted: controller.setPositions(selectedFile)
    }
    FolderDialog {
        id: outputDialog
        title: "Select output directory"
        onAccepted: controller.setOutputDir(selectedFolder)
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 12
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 14

            GroupBox {
                title: "Input"
                Layout.fillWidth: true
                GridLayout {
                    columns: 3
                    anchors.fill: parent
                    columnSpacing: 8

                    Label { text: "Edge list" }
                    TextField {
                        Layout.fillWidth: true
                        text: controller.edgeList
                        placeholderText: "…_edgelist.csv"
                        onEditingFinished: controller.setEdgeList(text)
                    }
                    Button { text: "Browse…"; onClicked: edgeDialog.open() }

                    Label { text: "Positions" }
                    TextField {
                        Layout.fillWidth: true
                        text: controller.positions
                        placeholderText: "…_positions.csv"
                        onEditingFinished: controller.setPositions(text)
                    }
                    Button { text: "Browse…"; onClicked: positionsDialog.open() }

                    Label { text: "Output dir" }
                    TextField {
                        Layout.fillWidth: true
                        text: controller.outputDir
                        onEditingFinished: controller.setOutputDir(text)
                    }
                    Button { text: "Browse…"; onClicked: outputDialog.open() }
                }
            }

            GroupBox {
                title: "Parameters"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    Repeater {
                        model: controller.fields

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: modelData.label
                                Layout.preferredWidth: 210
                                ToolTip.text: modelData.help
                                ToolTip.visible: modelData.help ? hover.hovered : false
                                HoverHandler { id: hover }
                            }

                            // Editors as siblings gated on `visible`, not a
                            // Loader with inline Components: those get a context
                            // where neither `controller` nor `modelData` resolves.
                            // QtQuick.Layouts skips invisible items, so only the
                            // relevant one takes space.
                            CheckBox {
                                visible: modelData.kind === "bool"
                                checked: controller.valueOf(modelData.id)
                                onToggled: controller.setValue(modelData.id, checked)
                            }

                            TextField {
                                visible: modelData.kind === "text"
                                Layout.fillWidth: true
                                text: controller.valueOf(modelData.id)
                                onEditingFinished: controller.setValue(modelData.id, text)
                            }

                            TextField {
                                visible: modelData.kind === "number"
                                    || modelData.kind === "integer"
                                Layout.fillWidth: true
                                text: controller.valueOf(modelData.id)
                                validator: DoubleValidator {}
                                onEditingFinished: controller.setValue(modelData.id, text)
                            }

                            RowLayout {
                                visible: modelData.kind === "size"
                                Layout.fillWidth: true
                                TextField {
                                    Layout.preferredWidth: 90
                                    // Bindings of invisible siblings still
                                    // evaluate, so guard the indexing: on a
                                    // scalar field this would be undefined.
                                    text: modelData.kind === "size"
                                        ? controller.valueOf(modelData.id)[0] : ""
                                    validator: IntValidator { bottom: 1 }
                                    onEditingFinished: controller.setSize(modelData.id, 0, text)
                                }
                                Label { text: "×" }
                                TextField {
                                    Layout.preferredWidth: 90
                                    // Bindings of invisible siblings still
                                    // evaluate, so guard the indexing: on a
                                    // scalar field this would be undefined.
                                    text: modelData.kind === "size"
                                        ? controller.valueOf(modelData.id)[1] : ""
                                    validator: IntValidator { bottom: 1 }
                                    onEditingFinished: controller.setSize(modelData.id, 1, text)
                                }
                                Item { Layout.fillWidth: true }
                            }
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Button {
                    text: controller.running ? "Cancel" : "Run"
                    highlighted: !controller.running
                    onClicked: controller.running ? controller.cancel() : controller.run()
                }
                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: controller.percent
                    indeterminate: controller.running && controller.percent <= 0
                }
                Label {
                    text: controller.running ? controller.percent.toFixed(0) + "%" : ""
                    Layout.preferredWidth: 44
                }
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: controller.status
                color: controller.failed ? "#c0392b" : palette.text
            }

            GroupBox {
                title: "Log"
                Layout.fillWidth: true
                Layout.preferredHeight: 220

                ScrollView {
                    anchors.fill: parent
                    TextArea {
                        id: logArea
                        readOnly: true
                        wrapMode: TextArea.NoWrap
                        font.family: "monospace"
                        font.pixelSize: 11
                        text: controller.logText
                        onTextChanged: cursorPosition = length
                    }
                }
            }
        }
    }
}
