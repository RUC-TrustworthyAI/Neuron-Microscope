"""Native collection, restart and handoff tests using a tiny local Qwen3 model."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from language.collection.__main__ import parser, request_for, run
from language.collection.capture import collect, index
from language.collection.corpus import prepare, read, write


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.set_num_threads(1)
        torch.manual_seed(7)
        texts = [(' A local catalog describes mathematics and city services. ' * (i % 3 + 2)) + str(i) for i in range(40)]
        raw = Tokenizer(models.BPE(unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        raw.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=300, special_tokens=['[UNK]', '[PAD]'],
                                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]')
        model_dir = self.root / 'tiny-qwen3'
        model_dir.mkdir()
        self.tokenizer.save_pretrained(model_dir)
        config = Qwen3Config(vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
                             num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, max_position_embeddings=128, pad_token_id=self.tokenizer.pad_token_id)
        self.model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        self.model.save_pretrained(model_dir)
        (self.root / 'source.jsonl').write_text(''.join(json.dumps({'text': text}) + '\n' for text in texts[:20]))
        pq.write_table(pa.Table.from_arrays([pa.array(texts[20:])], schema=pa.schema([pa.field('text', pa.string(), nullable=False)])),
                       self.root / 'source.parquet', row_group_size=3)
        self.args = parser().parse_args([
            '--model', str(model_dir), '--corpus', str(self.root / 'source.jsonl'), str(self.root / 'source.parquet'),
            '--neurons', 'L0_N0', 'L1_N3', 'L2_N5', '--tokens', '1027', '--max-document-tokens', '64',
            '--shard-tokens', '180', '--batch-size', '3', '--batch-tokens', '128',
            '--report-workers', '1', '--output', str(self.root / 'archive')])

    def runtime(self):
        from common import Runtime
        return Runtime(self.model, self.tokenizer, {'devices': [], 'threads': 1}, {'layer': 0, 'index': 0})

    def test_native_bits_reports_and_existing_interpretation_inputs(self):
        observed, lengths, handles = {}, [], []

        def input_lengths(module, args, kwargs):
            lengths.append(kwargs['attention_mask'].sum(dim=1).tolist())

        handles.append(self.model.model.register_forward_pre_hook(input_lengths, with_kwargs=True))
        for layer, neuron in [(0, 0), (1, 3), (2, 5)]:
            for channel in ('gate', 'value', 'product'):
                observed[layer, channel] = []

                def save(module, args, output=None, layer=layer, neuron=neuron, channel=channel):
                    tensor = args[0] if channel == 'product' else output
                    observed[layer, channel].append(tensor[:, :, neuron].detach().cpu().contiguous()
                                                    .view(torch.uint16).numpy().copy())

                mlp = self.model.model.layers[layer].mlp
                handles.append(mlp.down_proj.register_forward_pre_hook(save) if channel == 'product' else
                               getattr(mlp, 'gate_proj' if channel == 'gate' else 'up_proj').register_forward_hook(save))
        try:
            root = run(self.args, self.runtime())
        finally:
            for handle in handles:
                handle.remove()
        from dataset_reader import Dataset, floats
        from language.workflow.sampling import prepare as sample
        from language.workflow.evidence import spectrum_packet
        from language.examples.configure import configuration
        with Dataset(root) as archive:
            self.assertEqual(len(archive.token_ids), 1027)
            self.assertEqual(archive.manifest['channels'], ['gate', 'value', 'product'])
            self.assertEqual(int(archive.documents['token_count'].to_numpy().sum()), 1027)
            self.assertLess(int(archive.documents['token_count'][-1].as_py()), 64)
            for column, unit in enumerate(archive.neurons):
                actual_channels = np.concatenate([values for _, values in archive.iter_neuron(unit['id'])])
                for channel, name in enumerate(('gate', 'value', 'product')):
                    expected = np.concatenate([batch[row, :length] for batch, batch_lengths in
                                               zip(observed[unit['layer'], name], lengths)
                                               for row, length in enumerate(batch_lengths)])
                    np.testing.assert_array_equal(actual_channels[:, channel], floats(expected))
                actual = actual_channels[:, 2]
                self.assertEqual(int(archive.histogram(unit['id']).sum()), 1027)
                extrema = archive.extremes(unit['id'], k=3)
                self.assertEqual(extrema[0]['value'], float(actual.max()))
                self.assertEqual(len(archive.statistics(unit['id'])), 3)
                folder = root / 'reports' / unit['id']
                for name in ('index.html', 'report.json', 'dataset-index.json', 'token-rankings.json', 'effects.json'):
                    self.assertTrue((folder / name).is_file(), name)
                data = read(folder / 'report.json')
                self.assertEqual(data['identity']['model'], 'tiny-qwen3')
                for point in data['points'].values():
                    saved = archive.document_activations(int(point['document']), unit['id'])
                    np.testing.assert_array_equal(point['values'], saved['activations'][point['position']])
                config = configuration(unit['id'], self.args.model, folder, root, [0])
                output = self.root / ('sample-' + unit['id'])
                sample(folder, config, root, output)
                spectrum = read(output / 'sections.json')['spectrum']
                self.assertEqual(len(spectrum['points']), 500)
                packet = spectrum_packet(folder, spectrum, read(output / 'saved-contexts.json'))
                self.assertTrue(packet['logit_effect']['promote'])
                self.assertTrue(packet['logit_effect']['suppress'])
                with torch.no_grad():
                    vector = self.model.model.layers[unit['layer']].mlp.down_proj.weight[:, unit['neuron']].float()
                    direct = self.model.lm_head.weight.float() @ vector
                self.assertAlmostEqual(packet['logit_effect']['promote'][0]['weight'], float(direct.max()), places=6)
                self.assertAlmostEqual(packet['logit_effect']['suppress'][0]['weight'], float(direct.min()), places=6)
                self.assertEqual(read(output / 'sampling.json')['new_model_forwards'], 0)
        snapshot = hashlib.sha256((root / 'part-0/shard-000000.npy').read_bytes()).hexdigest()
        with patch.object(self.model.model, 'forward', side_effect=AssertionError('Completed collection was measured again')):
            run(self.args, self.runtime())
        self.assertEqual(snapshot, hashlib.sha256((root / 'part-0/shard-000000.npy').read_bytes()).hexdigest())
        (root / 'reports/complete.json').unlink()
        shutil.rmtree(root / 'reports/L2_N5')
        self.args.report_workers = 2
        with patch.object(self.model.model, 'forward', side_effect=AssertionError('Report recovery called the model')):
            run(self.args, self.runtime())
        self.assertTrue((root / 'reports/L2_N5/effects.json').is_file())

    def test_interrupted_capture_reuses_committed_shards(self):
        request, neurons = request_for(self.args)
        root = self.args.output
        root.mkdir()
        write(root / 'request.json', request)
        prepare(root, request, neurons)
        forward = self.model.model.forward
        def interrupt(*args, **kwargs):
            if (root / 'capture-state.json').exists():
                raise RuntimeError('Interrupted fixture')
            return forward(*args, **kwargs)

        with patch.object(self.model.model, 'forward', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'Interrupted fixture'):
            collect(root, self.model)
        committed = root / 'part-0/shard-000000.npy'
        self.assertEqual(read(root / 'capture-state.json')['completed_shards'], [0])
        timestamp = committed.stat().st_mtime_ns
        collect(root, self.model)
        index(root)
        self.assertTrue(read(root / 'verification.json')['complete'])
        self.assertEqual(committed.stat().st_mtime_ns, timestamp)
        self.assertFalse(any(self.model.model.layers[0].mlp.down_proj._forward_pre_hooks))

    def test_input_sampling_is_reproducible_and_budget_is_exact(self):
        request, neurons = request_for(self.args)
        outputs = []
        for name in ('one', 'two'):
            root = self.root / name
            root.mkdir()
            prepare(root, request, neurons)
            outputs.append(root)
        np.testing.assert_array_equal(np.load(outputs[0] / 'token_ids.npy'), np.load(outputs[1] / 'token_ids.npy'))
        self.assertEqual(read(outputs[0] / 'inputs/block-000000.documents.json'),
                         read(outputs[1] / 'inputs/block-000000.documents.json'))
        request['tokens'] = 100_000
        short = self.root / 'short'
        short.mkdir()
        with self.assertRaisesRegex(ValueError, 'Corpus supplied'):
            prepare(short, request, neurons)
        self.assertFalse((short / 'inputs-complete.json').exists())


if __name__ == '__main__':
    unittest.main()
