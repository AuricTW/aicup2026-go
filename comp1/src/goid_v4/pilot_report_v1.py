"""Generate the measured milestone report and immutable candidate references."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from .pilot_scope_v1 import ROOT,RUN,sha,write
from .pilot_outer_v1 import OUT
from .pilot_outer_heads_v2 import HEAD

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    summary=read(OUT/'outer_reveal_v1/SUMMARY.json');dev=read(RUN/'development_reveal_v1/SUMMARY.json')
    for name in ('scope_closure_audit.json','saved_output_review.json'):
        if read(OUT/name)['status']!='passed':raise RuntimeError('Closure audit not passed')
    if read(OUT/'postrun_exposure_and_episode_closure.json')['status']!='passed_postrun_closure_not_backdated':raise RuntimeError('Formal closure missing')
    measurements=[]
    for phase,root in [('development',RUN),('outer',HEAD)]:
        for path in sorted((root/'models').glob('*/result.json')):
            r=read(path)
            if 'training_elapsed_seconds' in r:
                measurements.append({'phase':phase,'method':r['method'],'seed':r['seed'],'steps':r['steps'],
                    'train_segment_seconds':r['training_elapsed_seconds'],'fit_calibration_included_seconds':r['total_elapsed_seconds'],
                    'peak_allocated_GiB':r['peak_allocated_GiB']})
    base=read(OUT/'baseline_refit_result.json')
    basepeak=max(r['peak_allocated_GiB'] for r in map(json.loads,(OUT/'base_training.jsonl').read_text(encoding='utf-8').splitlines()))
    accounting={'training_runs':measurements,'base_refit':base,'base_peak_allocated_GiB':basepeak,
        'total_successful_training_segment_seconds':sum(r['train_segment_seconds'] for r in measurements)+base['train_seconds'],
        'timing_scope':'elapsed training-loop segments on GPU jobs, includes CPU sampling; not kernel-only timing or complete task wall time',
        'peak_scope':'PyTorch allocated tensors inside individual measured jobs; excludes CUDA context/other processes/total device usage',
        'data_export_seconds':{'development':read(RUN/'data/scope_receipt.json')['elapsed_seconds'],'outer':read(OUT/'data/scope_receipt.json')['elapsed_seconds']},
        'no_external_compute':True,'financial_cost_not_measured':True,'local_resources':'local_resources.json'}
    write(OUT/'resource_accounting.json',accounting)
    records=read(OUT/'data/allowed_records.json');plans=read(OUT/'data/outer_plans.json')
    rank_by_player={r['player']:r['rank'] for r in records};bucket=defaultdict(list);errors=[]
    preds={r['question_id']:r for r in csv.DictReader((OUT/'outer_reveal_v1/U0_s31415/predictions.csv').open(encoding='utf-8'))}
    for pool in plans:
        b=np.load(OUT/'outer_reveal_v1'/pool['name']/'color_rrf.npy',allow_pickle=False)
        for i,q in enumerate(pool['questions']):
            row=preds[q['question_id']];same=all(records[j]['color']==records[q['indices'][0]]['color'] for j in q['indices'])
            bucket[q['q'],'single_color' if same else 'mixed_color'].append(row)
            baseline_player=pool['candidate_order'][int(np.argmax(b[i]))]
            errors.append({'q':q['q'],'baseline_wrong':int(row['color_rank'])!=1,'fused_wrong':int(row['fused_color_rank'])!=1,
                'baseline_top1_same_rank':rank_by_player[baseline_player]==rank_by_player[q['player']],
                'fused_top1_same_rank':rank_by_player[row['top1']]==rank_by_player[q['player']]})
    slices=[]
    for (q,color),rows in sorted(bucket.items()):
        slices.append({'q':q,'query_colors':color,'n':len(rows),'mean_score_gain':float(np.mean([float(r['color_score_gain']) for r in rows])),
            'fused_top1':float(np.mean([int(r['fused_color_rank'])==1 for r in rows])),'small_cell':len(rows)<20})
    wrongs=[]
    for q in (5,10,20):
        items=[r for r in errors if r['q']==q];out={'q':q}
        for method in ('baseline','fused'):
            bad=[r for r in items if r[f'{method}_wrong']]
            out[f'{method}_wrong']=len(bad);out[f'{method}_wrong_top1_same_rank']=sum(r[f'{method}_top1_same_rank'] for r in bad)
        wrongs.append(out)
    write(OUT/'descriptive_slices.json',{'status':'posthoc_descriptive_not_new_gates_or_routes','seed':31415,'color_slices':slices,'wrong_top1_rank_composition':wrongs})
    checkpoint=HEAD/'models/U0_mean_count_s31415/model.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    freeze={'status':'internal_candidate_frozen_G3_pending','version':'u0_mean_count_outer_v1_20261026','selected_seed':31415,
        'selection':'preselected before new outer scores; never best-of-three',
        'model_path':str(checkpoint.relative_to(ROOT)),'model_sha256':sha(checkpoint),
        'base_bundle_path':str((OUT/'base_bundle').relative_to(ROOT)),
        'base_file_sha256':read(OUT/'base_bundle/manifest.json')['files'],
        'head_trainable_parameters':sum(v.numel() for k,v in saved['state_dict'].items() if k!='prior_variance'),
        'floor':saved['floor'],'variance_scale':saved['variance_scale'],'head_steps':5000,
        'feature_dimension':1728,'embedding_dimension':128,'fusion_new_weight':.25,'rrf_k':60,
        'feature_source':'newly refitted statistics1582 + MLP128 + continuous12 + B/W2 + length1 + stages3',
        'numeric_policy':'FP32 GPU fixed1024 game encoder; fixed32x100 set encoder; FP32 CPU query GEMV; FP64 rank fusion; TF32/autocast disabled',
        'G2':summary['G2'],'inference_ready':False,'official_submission_authorized':False,
        'fallback':'existing scripts/run_inference_v3.py and original/color RRF unchanged',
        'rule_flags':{k:False for k in ('external_data','external_weights','candidate_finetuning','query_tta','cross_query_statistics','pseudolabel','joint_assignment')},
        'retired400_reused':False,'original_U2_gate_changed':False}
    write(OUT/'CANDIDATE_FREEZE.json',freeze)
    doc=ROOT/'docs/research_v4/p1_u0_outer_v1_20261026';doc.mkdir(exist_ok=True)
    table='\n'.join(f"| {r['q']} | {r['color_baseline']:.6f} | {r['fused_color_score']:.6f} | +{r['color_score_gain']:.6f} | {r['preselected_seed31415_score']:.6f} |" for r in summary['aggregates'])
    ci=summary['primary_seed31415_paired_CI'];citable='\n'.join(f'| {q} | {p:+.6f} | {lo:+.6f} | {threshold:+.3f} |' for q,p,lo,threshold in zip((5,10,20),ci['point'],ci['lower'],(0,0,-.005)))
    time_total=accounting['total_successful_training_segment_seconds']
    text=f'''# U0前瞻封鎖內部評估：實測里程碑

**mean＋count不確定性模型與顏色RRF的固定融合，在新800人外層評估通過事前G1數值重現及G2內部gate。原U2在第一輪失敗的結論保留。候選尚未接入完整CSV管線，G3未執行。**

本頁是最新實驗狀態。根目錄PROJECT_STATUS.json與所有歷史協議保持原樣，不用其舊phase推論本輪沒有訓練。

| 查詢局數 | 重新擬合顏色RRF | U0固定融合（三seed平均） | 增益 | 事前指定seed31415融合 |
|---:|---:|---:|---:|---:|
{table}

每個q有800位不同目標，分兩個互斥400候選池，每人100局support與巢狀5／10／20局query。三seed不是2400位不同玩家；上表三seed平均也不是把三模型ensemble。

主要不確定性只看事前固定seed31415。10,000次共享player-cluster bootstrap，family=3，一側覆蓋率精確為59/60；推論條件是固定兩個候選池。

| q | 指定seed分數增益 | 校正後增益下界 | 通過門檻（須嚴格大於） |
|---:|---:|---:|---:|
{citable}

G1三seed平均短局增益{(summary['aggregates'][0]['color_score_gain']+summary['aggregates'][1]['color_score_gain'])/2:.6f}，三seed短局增益都為正；q20及Top1防退步條件皆通過。G1程式回傳字串passed_development_only沿用既有數值檢查器名稱，本次資料route由獨立G2紀錄明定為prospective_blocked_internal，不因此升格為pristine。

## 證據適用範圍

這800人來自舊fit部分，曾參與**歷史**MLP擬合；在第一輪U0／U2的400人訓練、100人校準中均未使用。它們不等於原始未曝光玩家，也不是封存400人。這次先按metadata選定800人及兩個候選池，才重新擬合所有當前狀態；基線正規化及新MLP使用其餘3974人、336,737局，排除800人的**全部**原fit局。新U0以原400人48000局訓練、原100人12000局校準，但所有表示重建。

新scope、既有fit身份公式及raw／move／canonical／effective-group交集在訓練前檢查；關閉舊權重載入。收尾稽核又由新背景資料完整重算正規化統計，與保存參數逐位一致，並核對新MLP SHA不同於原模型。原始混合來源只有凍結fit／既有development指定列的主鍵SEARCH與索引讀取，沒有全掃或全檔重hash。

**這是支持互補性的前瞻封鎖內部證據，並非官方成績、獨立pristine確認或名次保證。**歷史模型／方法選擇曝光不能因重訓而抹除；未知編輯棋譜、短前綴、對手及時間關聯也未全部排除。兩池是≥120局活躍玩家與固定棋力配額，不能把本表絕對分數直接和另一資料人口的development表比較。

800人現已分數與錯誤曝光；後續選模不能再稱其未見。formal AllowedGame／六組巢狀episode閉合檢查在收尾補做，明確標為postrun而非補寫為事前檢查；訓練前用的是相應metadata／玩家／群組隔離檢查。

## 方法及固定規則

每局1728D輸入依序為1582統計、128D重擬合MLP、12摘要、2色、1長度與3階段。U0每局投影至128D、遮罩平均，再加5維局數／黑白局數／缺色特徵，預測128D單位均值與對角變異。trainable參數{freeze['head_trainable_parameters']:,}。支援100局與query5／10／20局的Gaussian分數以固定rank60融合：0.75×顏色RRF＋0.25／(60＋U0排名)。未調alpha、按棋力route、挑最佳seed或挑checkpoint。

每seed5000更新、最後checkpoint；AdamW lr.001、wd.0001、clip5、dropout.1。每批兩種棋力各16人，同棋力負樣本15個與跨棋力15個，query局數機率.4／.4／.2。variance floor／scale只從fit-inner校準；輸入無玩家ID、棋力標籤或GT（棋力僅用於fit批次抽樣）。

GPU遊戲編碼固定1024筆，集合固定32×100並丟棄補齊輸出；FP32、不用TF32／autocast；CPU prototype投影每題GEMV，RRF與最終Gaussian分數FP64。集合、prototype均依content-key排序。重新擬合正規化採float64累積／方差後存float32；相對舊fit實作的mean先轉float32再算方差有微小公式差異，未宣稱與舊參數或舊中間矩陣相同。所有數值保證限本機固定stack。

## 失敗、稽核與資源

- 第一輪主候選ISAB的5／10／20局融合退步，G1失敗；U0正向對照才啟動本輪新版本，沒有改判原實驗。U1缺色修復及初始測試錯誤都有原紀錄。
- 本輪第一次head啟動因effective_group_token欄位缺少，在首個optimizer step前停止。heads_v2只新增既有group的別名，原失敗目錄／原匯出／原trainer保持，特徵bytes與玩家未變。
- 現有140個受保護程式／設定／文件、8個釘選development metadata、原bundle三個安全檔SHA核對未變。此140項不是全專案1450檔的重掃聲明；封存明細檔未為hash驗證而讀取。
- 既有62項CPU契約重跑通過，新增pilot10項通過。第一輪42,000列保存預測核對通過；本輪7200列以獨立Python穩定排序／rank／指數分數檢查，並逐列重算400候選固定融合與基線RRF，共12,000列矩陣檢查通過，沒有新模型forward。
- i7-12700、32GB RAM、RTX3060 12GB，全程本機。兩輪共9個集合訓練＋1個新MLP，各5000更新；成功訓練區段經過時間合計約{time_total/60:.2f}分鐘。這包含CPU抽樣，**不是純GPU kernel或整個任務wall time**。U2每seed約4.5分鐘，第一輪U0約40–45秒，本輪U0約111–113秒；不能把跨run耗時差異直接歸因模型品質。
- 已記錄PyTorch峰值allocated：U2約0.640GiB、U0約0.577GiB、重擬合MLP約{basepeak:.3f}GiB；均非總VRAM／CUDA context。沒有租用算力，電費與本機折舊未量測，不報虛構金額。

## 產物與下一階段

主要原始結果：runs/research_v4/p1_u0_outer_v1_20261026/outer_reveal_v1/SUMMARY.json。CANDIDATE_FREEZE.json固定事前seed及基線／head SHA；resource_accounting.json、scope_closure_audit.json、saved_output_review.json、postrun_exposure_and_episode_closure.json分別保存資源、隔離、數值與曝光證據。descriptive_slices.json只作事後錯誤描述，不是新增gate或route。

下一階段優先完成獨立v4 U0候選CSV入口、模型包及v3回退：輸入候選／query CSV，限制400候選／100局support及正式schema，runtime不得讀audit、GT、舊混合矩陣或封存細節；game／set固定批次規則可重現，驗證冷暖重跑、子集／排序及保存矩陣一致性。僅用fit-inner或既有development做工程對齊，不重新把已曝光800人稱為新確認集。G3以前不宣稱可提交。

目前暫緩大棋盤模型與額外算力：低成本聚合已提供可重現互補增益，應先可靠整合。P2／P3仍是後續假設，若再研究須新版本scope／曝光盤點與預先門檻；不能因本輪成功直接承諾其收益。原v3、原RRF及顏色RRF均可回退；原封存400人未讀／未評分，沒有官方送件或外部資料／權重。本輪未新增網路文獻研究；新增實作由Codex依既有審核方案撰寫。
'''
    with (doc/'REPORT.md').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    status={'updated_at':now,'phase':'P1_real_pilot_and_U0_prospective_internal_complete_G3_pending',
        'original_U2_primary_G1':'failed','new_U0_G2':summary['G2'],'pristine_confirmation':False,
        'official_score':False,'submission':False,'retired400_read_or_evaluated':False,
        'latest_report':str((doc/'REPORT.md').relative_to(ROOT)),
        'candidate_freeze':str((OUT/'CANDIDATE_FREEZE.json').relative_to(ROOT)),
        'G3':'pending_CSV_end_to_end_integration_and_reproducibility','next':'v4_U0_CSV_pipeline_with_v3_fallback',
        'larger_board_or_compute':'deferred_pending_reliable_U0_integration',
        'root_PROJECT_STATUS_and_historical_protocols':'preserved; read this versioned status for current experiments'}
    write(doc/'STATUS.json',status)
    write(ROOT/'docs/research_v4/EXPERIMENT_STATUS_v1_20261026.json',status)
    write(RUN/'milestone_closure.json',{'first_pilot_source_and_results_preserved':True,'next_stage_completed':str(OUT.relative_to(ROOT)),
        'original_U2_gate_still_failed':True,'U0_was_control_not_relabelled_in_first_pilot':True,'latest_status':str((doc/'STATUS.json').relative_to(ROOT))})
    print(json.dumps({'report':str(doc/'REPORT.md'),'training_segments_minutes':time_total/60,'candidate':freeze['status']}),flush=True)

if __name__=='__main__':main()
